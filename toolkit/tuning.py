"""
Auto-tuning functions for SNN hyperparameters
Includes learning rates, thresholds, surrogate gradients optimization
"""

import numpy as np
from typing import Dict, List, Tuple, Callable, Optional, Any
from dataclasses import dataclass
import json
import time

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError:
    torch = None
    nn = None
    optim = None


@dataclass
class TuningConfig:
    """Configuration for hyperparameter tuning"""
    param_ranges: Dict[str, Tuple[float, float]]
    num_trials: int = 50
    num_epochs: int = 10
    early_stopping_patience: int = 5
    optimization_metric: str = "accuracy"  # "accuracy", "loss", "spike_rate"
    minimize_metric: bool = False
    random_seed: Optional[int] = None


class AutoTuner:
    """
    Automated hyperparameter tuning for SNN models
    Supports grid search, random search, and basic Bayesian optimization
    """
    
    def __init__(
        self,
        model_builder: Callable,
        train_data: Any,
        val_data: Any,
        config: TuningConfig
    ):
        self.model_builder = model_builder
        self.train_data = train_data
        self.val_data = val_data
        self.config = config
        
        self.trial_history = []
        self.best_params = None
        self.best_score = float('inf') if config.minimize_metric else float('-inf')
        
        if config.random_seed:
            np.random.seed(config.random_seed)
    
    def random_search(self) -> Dict[str, Any]:
        """
        Perform random search over hyperparameter space
        
        Returns:
            Dictionary containing best parameters and results
        """
        print(f"Starting random search with {self.config.num_trials} trials...")
        
        for trial in range(self.config.num_trials):
            # Sample random parameters
            params = self._sample_random_params()
            
            # Train and evaluate model
            score, metrics = self._evaluate_params(params, trial)
            
            # Track best parameters
            if self._is_better_score(score):
                self.best_score = score
                self.best_params = params.copy()
                print(f"Trial {trial}: New best {self.config.optimization_metric} = {score:.4f}")
            else:
                print(f"Trial {trial}: {self.config.optimization_metric} = {score:.4f}")
            
            # Store trial results
            self.trial_history.append({
                'trial': trial,
                'params': params,
                'score': score,
                'metrics': metrics
            })
        
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'trial_history': self.trial_history
        }
    
    def grid_search(self, grid_resolution: int = 5) -> Dict[str, Any]:
        """
        Perform grid search over hyperparameter space
        
        Args:
            grid_resolution: Number of points per parameter dimension
            
        Returns:
            Dictionary containing best parameters and results
        """
        print(f"Starting grid search with resolution {grid_resolution}...")
        
        # Generate grid points
        param_grids = {}
        for param, (min_val, max_val) in self.config.param_ranges.items():
            param_grids[param] = np.linspace(min_val, max_val, grid_resolution)
        
        # Generate all combinations
        param_combinations = self._generate_param_combinations(param_grids)
        total_trials = len(param_combinations)
        
        print(f"Total combinations to evaluate: {total_trials}")
        
        for trial, params in enumerate(param_combinations):
            # Train and evaluate model
            score, metrics = self._evaluate_params(params, trial)
            
            # Track best parameters
            if self._is_better_score(score):
                self.best_score = score
                self.best_params = params.copy()
                print(f"Trial {trial}: New best {self.config.optimization_metric} = {score:.4f}")
            
            # Store trial results
            self.trial_history.append({
                'trial': trial,
                'params': params,
                'score': score,
                'metrics': metrics
            })
        
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'trial_history': self.trial_history
        }
    
    def _sample_random_params(self) -> Dict[str, float]:
        """Sample random parameters from specified ranges"""
        params = {}
        for param, (min_val, max_val) in self.config.param_ranges.items():
            if param in ['batch_size', 'hidden_size']:
                # Integer parameters
                params[param] = int(np.random.uniform(min_val, max_val))
            else:
                # Float parameters
                if param.endswith('_rate') or param == 'threshold':
                    # Log scale for learning rates and small thresholds
                    log_min = np.log10(min_val)
                    log_max = np.log10(max_val)
                    params[param] = 10 ** np.random.uniform(log_min, log_max)
                else:
                    params[param] = np.random.uniform(min_val, max_val)
        return params
    
    def _generate_param_combinations(self, param_grids: Dict[str, np.ndarray]) -> List[Dict[str, float]]:
        """Generate all combinations of grid parameters"""
        import itertools
        
        keys = list(param_grids.keys())
        values = list(param_grids.values())
        
        combinations = []
        for combination in itertools.product(*values):
            param_dict = dict(zip(keys, combination))
            combinations.append(param_dict)
        
        return combinations
    
    def _evaluate_params(self, params: Dict[str, Any], trial: int) -> Tuple[float, Dict[str, Any]]:
        """
        Evaluate a set of parameters by training a model
        
        Returns:
            Tuple of (score, metrics_dict)
        """
        try:
            # Build model with current parameters
            model = self.model_builder(**params)
            
            # Train model
            metrics = self._train_model(model, params)
            
            # Extract optimization score
            score = metrics.get(self.config.optimization_metric, 0.0)
            
            return score, metrics
            
        except Exception as e:
            print(f"Trial {trial} failed with error: {str(e)}")
            # Return worst possible score for failed trials
            return float('inf') if self.config.minimize_metric else float('-inf'), {}
    
    def _train_model(self, model: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Train the model and return evaluation metrics
        This is a simplified training loop - should be customized for specific needs
        """
        if torch is None:
            raise ImportError("PyTorch is required for training")
        
        # Setup training
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=params.get('learning_rate', 0.001))
        criterion = nn.CrossEntropyLoss()
        
        best_val_accuracy = 0.0
        patience_counter = 0
        
        # Training loop
        for epoch in range(self.config.num_epochs):
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            # Training phase (simplified - assumes DataLoader format)
            for batch_idx, (data, targets) in enumerate(self.train_data):
                if batch_idx >= 10:  # Limit training for tuning speed
                    break
                    
                data, targets = data.to(device), targets.to(device)
                
                optimizer.zero_grad()
                outputs = model(data)
                
                # Handle time-series output (take final time step)
                if outputs.dim() > 2:
                    outputs = outputs[:, -1, :]
                
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += targets.size(0)
                train_correct += predicted.eq(targets).sum().item()
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            total_spikes = 0
            
            with torch.no_grad():
                for batch_idx, (data, targets) in enumerate(self.val_data):
                    if batch_idx >= 5:  # Limit validation for speed
                        break
                        
                    data, targets = data.to(device), targets.to(device)
                    outputs = model(data, record_spikes=True)
                    
                    # Handle time-series output
                    if outputs.dim() > 2:
                        outputs = outputs[:, -1, :]
                    
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
                    
                    _, predicted = outputs.max(1)
                    val_total += targets.size(0)
                    val_correct += predicted.eq(targets).sum().item()
                    
                    # Count spikes for spike rate metric
                    if hasattr(model, 'get_spike_counts'):
                        try:
                            spike_counts = model.get_spike_counts()
                            total_spikes += sum(counts.sum().item() for counts in spike_counts)
                        except:
                            pass
            
            val_accuracy = val_correct / val_total if val_total > 0 else 0.0
            
            # Early stopping
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config.early_stopping_patience:
                    break
        
        # Calculate final metrics
        spike_rate = total_spikes / (val_total * model.layers[-1].output_size) if val_total > 0 else 0.0
        
        metrics = {
            'accuracy': best_val_accuracy,
            'loss': val_loss / max(len(self.val_data), 1),
            'spike_rate': spike_rate,
            'epochs_trained': epoch + 1
        }
        
        return metrics
    
    def _is_better_score(self, score: float) -> bool:
        """Check if current score is better than best score"""
        if self.config.minimize_metric:
            return score < self.best_score
        else:
            return score > self.best_score
    
    def save_results(self, filename: str):
        """Save tuning results to JSON file"""
        results = {
            'config': {
                'param_ranges': self.config.param_ranges,
                'num_trials': self.config.num_trials,
                'optimization_metric': self.config.optimization_metric,
                'minimize_metric': self.config.minimize_metric
            },
            'best_params': self.best_params,
            'best_score': self.best_score,
            'trial_history': self.trial_history
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {filename}")
    
    def load_results(self, filename: str):
        """Load previously saved tuning results"""
        with open(filename, 'r') as f:
            results = json.load(f)
        
        self.best_params = results['best_params']
        self.best_score = results['best_score']
        self.trial_history = results['trial_history']
        
        print(f"Results loaded from {filename}")


class SurrogateGradientTuner:
    """
    Specialized tuner for surrogate gradient parameters in SNN training
    """
    
    def __init__(self):
        self.available_surrogates = [
            'rectangular',
            'triangular',
            'arctangent',
            'sigmoid',
            'fast_sigmoid'
        ]
    
    def tune_surrogate_params(
        self,
        model_builder: Callable,
        train_data: Any,
        val_data: Any,
        num_trials: int = 20
    ) -> Dict[str, Any]:
        """
        Tune surrogate gradient function and parameters
        
        Returns:
            Best surrogate configuration
        """
        best_config = None
        best_score = float('-inf')
        results = []
        
        for trial in range(num_trials):
            # Sample surrogate configuration
            surrogate_type = np.random.choice(self.available_surrogates)
            alpha = np.random.uniform(0.1, 10.0)  # Surrogate sharpness
            
            config = {
                'surrogate_type': surrogate_type,
                'alpha': alpha
            }
            
            try:
                # Build and train model
                model = model_builder(surrogate_config=config)
                score = self._evaluate_surrogate(model, train_data, val_data)
                
                if score > best_score:
                    best_score = score
                    best_config = config
                
                results.append({
                    'config': config,
                    'score': score
                })
                
                print(f"Trial {trial}: {surrogate_type} (α={alpha:.2f}) → {score:.4f}")
                
            except Exception as e:
                print(f"Trial {trial} failed: {str(e)}")
        
        return {
            'best_config': best_config,
            'best_score': best_score,
            'all_results': results
        }
    
    def _evaluate_surrogate(self, model: Any, train_data: Any, val_data: Any) -> float:
        """Quick evaluation of surrogate gradient configuration"""
        # Simplified evaluation - train for few epochs and return validation accuracy
        if torch is None:
            return 0.0
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # Quick training
        model.train()
        for epoch in range(3):  # Very short training
            for batch_idx, (data, targets) in enumerate(train_data):
                if batch_idx >= 5:  # Only few batches
                    break
                    
                data, targets = data.to(device), targets.to(device)
                optimizer.zero_grad()
                
                outputs = model(data)
                if outputs.dim() > 2:
                    outputs = outputs[:, -1, :]
                
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
        
        # Quick validation
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(val_data):
                if batch_idx >= 3:  # Only few batches
                    break
                    
                data, targets = data.to(device), targets.to(device)
                outputs = model(data)
                
                if outputs.dim() > 2:
                    outputs = outputs[:, -1, :]
                
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        return correct / total if total > 0 else 0.0


def create_default_tuning_config() -> TuningConfig:
    """Create a default tuning configuration for common SNN parameters"""
    return TuningConfig(
        param_ranges={
            'learning_rate': (1e-5, 1e-2),
            'threshold': (0.1, 2.0),
            'decay': (0.8, 0.99),
            'hidden_size': (64, 512),
            'batch_size': (16, 128)
        },
        num_trials=30,
        num_epochs=5,
        optimization_metric="accuracy",
        minimize_metric=False
    )