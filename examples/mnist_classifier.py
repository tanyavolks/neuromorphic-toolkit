#!/usr/bin/env python3
"""
MNIST Classifier Demo using SNN Toolkit

This example demonstrates how to build, train, and evaluate a spiking neural network
for MNIST digit classification using the neuromorphic SNN toolkit.
"""

import sys
import os

# Add parent directory to path for toolkit imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any

from toolkit import SNNModel, SNNLayer, load_dataset
from toolkit.utils import save_model_checkpoint, calculate_spike_metrics

# Import optional components with fallbacks
try:
    from toolkit import SpikeRasterPlot, AutoTuner
except ImportError:
    SpikeRasterPlot = AutoTuner = None
    print("Warning: Visualization and tuning components not available")

try:
    from toolkit.tuning import TuningConfig, create_default_tuning_config
except ImportError:
    TuningConfig = create_default_tuning_config = None
    print("Warning: Tuning components not available")


class MNISTSNNClassifier:
    """
    Complete MNIST classifier using Spiking Neural Networks
    """
    
    def __init__(
        self,
        hidden_size: int = 128,
        num_classes: int = 10,
        time_steps: int = 100,
        backend: str = "spiking_jelly",
        device: str = "auto"
    ):
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.time_steps = time_steps
        self.backend = backend
        self.device = device
        
        # Build network architecture
        self.architecture = [
            {
                'input_size': 784,  # 28x28 MNIST images
                'output_size': hidden_size,
                'neuron_type': 'LIF',
                'threshold': 1.0,
                'decay': 0.9
            },
            {
                'input_size': hidden_size,
                'output_size': hidden_size // 2,
                'neuron_type': 'LIF',
                'threshold': 1.0,
                'decay': 0.9
            },
            {
                'input_size': hidden_size // 2,
                'output_size': num_classes,
                'neuron_type': 'LIF',
                'threshold': 1.0,
                'decay': 0.9
            }
        ]
        
        # Initialize model
        self.model = None
        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss()
        
        # Training history
        self.train_history = {
            'epoch': [],
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'spike_rate': []
        }
    
    def build_model(self):
        """Build the SNN model"""
        print("Building SNN model...")
        
        if SNNModel is None:
            raise ImportError("SNNModel not available - check toolkit imports")
        
        self.model = SNNModel(
            architecture=self.architecture,
            backend=self.backend,
            device=self.device
        )
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        print(f"Model built with {sum(p.numel() for p in self.model.parameters())} parameters")
        return self.model
    
    def train_epoch(self, train_loader, epoch: int, verbose: bool = True):
        """Train for one epoch"""
        if self.model is None or self.optimizer is None:
            raise RuntimeError("Model or optimizer not initialized")
            
        self.model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        total_spikes = 0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(self.model.device), targets.to(self.model.device)
            
            # Reset model state
            if hasattr(self.model, 'reset_state'):
                self.model.reset_state()
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(data, record_spikes=True)
            
            # Take final time step output for classification
            if outputs.dim() > 2:
                final_output = outputs[:, -1, :]
            else:
                final_output = outputs
            
            # Calculate loss and backpropagate
            loss = self.criterion(final_output, targets)
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            _, predicted = final_output.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Count spikes
            try:
                spike_counts = self.model.get_spike_counts()
                batch_spikes = sum(counts.sum().item() for counts in spike_counts)
                total_spikes += batch_spikes
            except:
                pass
            
            if verbose and batch_idx % 50 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}: '
                      f'Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%')
        
        avg_loss = train_loss / len(train_loader)
        accuracy = 100. * correct / total
        avg_spike_rate = total_spikes / (total * self.architecture[-1]['output_size'])
        
        return avg_loss, accuracy, avg_spike_rate
    
    def evaluate(self, test_loader, verbose: bool = True):
        """Evaluate model on test set"""
        self.model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        total_spikes = 0
        
        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(self.model.device), targets.to(self.model.device)
                
                # Reset model state
                self.model.reset_state()
                
                # Forward pass
                outputs = self.model(data, record_spikes=True)
                
                # Take final time step output
                if outputs.dim() > 2:
                    final_output = outputs[:, -1, :]
                else:
                    final_output = outputs
                
                # Calculate loss
                loss = self.criterion(final_output, targets)
                test_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = final_output.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # Count spikes
                try:
                    spike_counts = self.model.get_spike_counts()
                    batch_spikes = sum(counts.sum().item() for counts in spike_counts)
                    total_spikes += batch_spikes
                except:
                    pass
        
        avg_loss = test_loss / len(test_loader)
        accuracy = 100. * correct / total
        avg_spike_rate = total_spikes / (total * self.architecture[-1]['output_size'])
        
        if verbose:
            print(f'Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.2f}%, '
                  f'Spike Rate: {avg_spike_rate:.4f}')
        
        return avg_loss, accuracy, avg_spike_rate
    
    def train(
        self,
        train_loader,
        test_loader,
        num_epochs: int = 10,
        save_path: str = None,
        visualize: bool = False
    ):
        """Full training loop"""
        print(f"Starting training for {num_epochs} epochs...")
        
        # Setup visualization if requested
        raster_plot = None
        if visualize:
            raster_plot = SpikeRasterPlot(
                num_neurons=self.architecture[-1]['output_size'],
                time_window=self.time_steps
            )
            raster_plot.start_realtime()
        
        best_accuracy = 0.0
        
        for epoch in range(num_epochs):
            print(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")
            
            # Train
            train_loss, train_acc, train_spike_rate = self.train_epoch(
                train_loader, epoch + 1
            )
            
            # Evaluate
            val_loss, val_acc, val_spike_rate = self.evaluate(test_loader)
            
            # Update history
            self.train_history['epoch'].append(epoch + 1)
            self.train_history['train_loss'].append(train_loss)
            self.train_history['train_acc'].append(train_acc)
            self.train_history['val_loss'].append(val_loss)
            self.train_history['val_acc'].append(val_acc)
            self.train_history['spike_rate'].append(val_spike_rate)
            
            # Save best model
            if val_acc > best_accuracy:
                best_accuracy = val_acc
                if save_path:
                    save_model_checkpoint(
                        self.model, self.optimizer, epoch + 1,
                        val_loss, val_acc, save_path
                    )
            
            # Update visualization
            if raster_plot and hasattr(self.model, 'spike_recordings'):
                try:
                    if self.model.spike_recordings and self.model.spike_recordings[-1]:
                        last_spikes = self.model.spike_recordings[-1][-1]  # Last layer, last time step
                        raster_plot.add_spikes(last_spikes[0].numpy())  # First batch item
                except:
                    pass
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"Spike Rate: {val_spike_rate:.4f}")
        
        if raster_plot:
            raster_plot.stop_realtime()
        
        print(f"\nTraining completed! Best validation accuracy: {best_accuracy:.2f}%")
        return self.train_history
    
    def plot_training_history(self):
        """Plot training curves"""
        if not self.train_history['epoch']:
            print("No training history to plot")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = self.train_history['epoch']
        
        # Loss curves
        ax1.plot(epochs, self.train_history['train_loss'], 'b-', label='Train Loss')
        ax1.plot(epochs, self.train_history['val_loss'], 'r-', label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy curves
        ax2.plot(epochs, self.train_history['train_acc'], 'b-', label='Train Acc')
        ax2.plot(epochs, self.train_history['val_acc'], 'r-', label='Val Acc')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        # Spike rate
        ax3.plot(epochs, self.train_history['spike_rate'], 'g-', label='Spike Rate')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Average Spike Rate')
        ax3.set_title('Network Spike Rate Over Training')
        ax3.legend()
        ax3.grid(True)
        
        # Training progress
        ax4.plot(epochs, self.train_history['val_acc'], 'r-', linewidth=2)
        ax4.fill_between(epochs, self.train_history['val_acc'], alpha=0.3, color='red')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Validation Accuracy (%)')
        ax4.set_title('Validation Accuracy Progress')
        ax4.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def demo_spike_visualization(self, test_loader):
        """Demonstrate spike visualization capabilities"""
        print("Demonstrating spike visualization...")
        
        # Get a batch of test data
        data, targets = next(iter(test_loader))
        data, targets = data.to(self.model.device), targets.to(self.model.device)
        
        # Take only first few samples for visualization
        data = data[:4]
        targets = targets[:4]
        
        # Create visualization
        raster_plot = SpikeRasterPlot(
            num_neurons=self.architecture[0]['output_size'],  # First hidden layer
            time_window=self.time_steps,
            figsize=(15, 8)
        )
        
        # Forward pass with spike recording
        self.model.eval()
        self.model.reset_state()
        
        with torch.no_grad():
            outputs = self.model(data, record_spikes=True)
        
        # Add spike data to visualization
        if hasattr(self.model, 'spike_recordings') and self.model.spike_recordings:
            for t, time_spikes in enumerate(self.model.spike_recordings[0]):  # First layer
                # Average over batch for visualization
                avg_spikes = time_spikes.mean(dim=0)
                raster_plot.add_spikes(avg_spikes.numpy(), t)
        
        # Update and show plot
        raster_plot.update_plot(0)
        plt.title(f'Spike Raster Plot - MNIST Samples (Labels: {targets.cpu().numpy()})')
        plt.show()
        
        # Calculate and display spike metrics
        if hasattr(self.model, 'spike_recordings') and self.model.spike_recordings:
            spike_tensor = torch.stack(self.model.spike_recordings[0])  # Shape: (time, batch, neurons)
            metrics = calculate_spike_metrics(spike_tensor)
            
            print("\nSpike Metrics:")
            for key, value in metrics.items():
                print(f"  {key}: {value:.4f}")


def run_hyperparameter_tuning(train_loader, test_loader):
    """Demonstrate automatic hyperparameter tuning"""
    print("\n" + "="*50)
    print("HYPERPARAMETER TUNING DEMO")
    print("="*50)
    
    def model_builder(**params):
        """Model builder function for tuning"""
        classifier = MNISTSNNClassifier(
            hidden_size=int(params.get('hidden_size', 128)),
            time_steps=100,
            backend="spiking_jelly"
        )
        model = classifier.build_model()
        return model
    
    # Create tuning configuration
    config = TuningConfig(
        param_ranges={
            'hidden_size': (64, 256),
            'learning_rate': (0.0001, 0.01),
            'threshold': (0.5, 2.0),
            'decay': (0.8, 0.95)
        },
        num_trials=5,  # Small number for demo
        num_epochs=3,
        optimization_metric="accuracy",
        minimize_metric=False
    )
    
    # Run tuning
    tuner = AutoTuner(model_builder, train_loader, test_loader, config)
    results = tuner.random_search()
    
    print(f"\nBest parameters found:")
    for param, value in results['best_params'].items():
        print(f"  {param}: {value}")
    print(f"Best accuracy: {results['best_score']:.4f}")
    
    return results


def main():
    """Main demo function"""
    print("="*60)
    print("NEUROMORPHIC SNN TOOLKIT - MNIST CLASSIFIER DEMO")
    print("="*60)
    
    # Configuration
    batch_size = 32
    num_epochs = 5
    time_steps = 100
    
    try:
        # Load dataset
        print("\n1. Loading MNIST dataset...")
        train_loader, test_loader = load_dataset(
            dataset_name="mnist",
            batch_size=batch_size,
            time_steps=time_steps,
            encoding="rate"
        )
        
        # Create classifier
        print("\n2. Building SNN classifier...")
        classifier = MNISTSNNClassifier(
            hidden_size=128,
            time_steps=time_steps,
            backend="spiking_jelly"
        )
        
        model = classifier.build_model()
        
        # Train model
        print("\n3. Training model...")
        history = classifier.train(
            train_loader,
            test_loader,
            num_epochs=num_epochs,
            save_path="mnist_snn_best.pth",
            visualize=False  # Set to True for real-time visualization
        )
        
        # Final evaluation
        print("\n4. Final evaluation...")
        final_loss, final_acc, final_spike_rate = classifier.evaluate(test_loader)
        
        # Plot results
        print("\n5. Plotting training history...")
        classifier.plot_training_history()
        
        # Demonstrate spike visualization
        print("\n6. Spike visualization demo...")
        classifier.demo_spike_visualization(test_loader)
        
        # Hyperparameter tuning demo (optional)
        print("\n7. Would you like to run hyperparameter tuning demo? (y/n)")
        # For automated demo, we'll skip this
        # response = input().lower()
        # if response == 'y':
        #     tuning_results = run_hyperparameter_tuning(train_loader, test_loader)
        
        print("\n" + "="*60)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print(f"Final Test Accuracy: {final_acc:.2f}%")
        print(f"Final Spike Rate: {final_spike_rate:.4f}")
        print("="*60)
        
    except ImportError as e:
        print(f"Missing dependencies: {e}")
        print("Please install required packages:")
        print("pip install torch torchvision spikingjelly snntorch matplotlib numpy")
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()