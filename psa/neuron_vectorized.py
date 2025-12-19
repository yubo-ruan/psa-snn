"""
Vectorized PSA Neurons - GPU-efficient implementation.

Key changes from neuron.py:
1. No Python loops over neurons
2. All operations are batched tensor ops
3. Neighbor predictions use sparse matrix ops
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
import math


class VectorizedPSALayer(nn.Module):
    """
    Fully vectorized PSA layer.

    All neurons processed in parallel with batched operations.
    Neighbor connectivity encoded as sparse adjacency.
    """

    def __init__(
        self,
        num_neurons: int,
        input_dim: int,
        neighbor_radius: int = 5,
        beta: float = 0.9,
        threshold: float = 1.0,
        prediction_window: int = 5,
        fast_lr: float = 0.01,
        slow_lr: float = 0.001,
        consolidation_rate: float = 0.001,
    ):
        super().__init__()
        self.num_neurons = num_neurons
        self.input_dim = input_dim
        self.neighbor_radius = neighbor_radius
        self.beta = beta
        self.prediction_window = prediction_window
        self.fast_lr = fast_lr
        self.slow_lr = slow_lr
        self.consolidation_rate = consolidation_rate

        # Learnable threshold per neuron (for homeostasis)
        self.threshold = nn.Parameter(torch.ones(num_neurons) * threshold)

        # Input weights [num_neurons, input_dim]
        self.input_weights = nn.Parameter(torch.randn(num_neurons, input_dim) * 0.1)

        # Build neighbor adjacency matrix [num_neurons, num_neurons]
        # Entry (i,j) = 1 if j is a neighbor of i
        self.num_neighbors = min(2 * neighbor_radius, num_neurons - 1)
        neighbor_adj = torch.zeros(num_neurons, num_neurons)
        for i in range(num_neurons):
            for offset in range(-neighbor_radius, neighbor_radius + 1):
                if offset != 0:
                    j = (i + offset) % num_neurons
                    neighbor_adj[i, j] = 1.0
        self.register_buffer('neighbor_adj', neighbor_adj)

        # Prediction weights: each neuron predicts all its neighbors
        # Shape: [num_neurons, num_neurons] but masked by neighbor_adj
        # We use dense matrix but mask during forward
        self.fast_pred_weights = nn.Parameter(torch.randn(num_neurons, num_neurons) * 0.1)
        self.register_buffer('slow_pred_weights', torch.zeros(num_neurons, num_neurons))
        self.pred_bias = nn.Parameter(torch.zeros(num_neurons, num_neurons))

        # State buffers
        self.register_buffer('membrane', torch.zeros(num_neurons))
        self.register_buffer('spike_history', torch.zeros(num_neurons, prediction_window))

        # Utility tracking
        self.register_buffer('synapse_utility', torch.zeros(num_neurons, num_neurons))
        self.register_buffer('synapse_age', torch.zeros(num_neurons, num_neurons))

        # Error history for surprise computation
        self.register_buffer('error_history', torch.zeros(100))
        self.error_idx = 0

    def reset(self):
        """Reset all state."""
        self.membrane.zero_()
        self.spike_history.zero_()
        self.error_history.zero_()
        self.error_idx = 0

    def forward(
        self,
        x: torch.Tensor,
        return_predictions: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict]:
        """
        Vectorized forward pass.

        Args:
            x: Input [batch, input_dim] or [input_dim]
            return_predictions: Whether to compute neighbor predictions

        Returns:
            spikes: Output spikes [batch, num_neurons] or [num_neurons]
            predictions: Predicted neighbor rates [num_neurons, num_neurons] (masked)
            info: Dict with surprise, etc.
        """
        squeeze = x.dim() == 1
        if squeeze:
            x = x.unsqueeze(0)
        batch_size = x.shape[0]
        device = x.device

        # Compute input currents: [batch, num_neurons]
        currents = F.linear(x, self.input_weights)

        # For batch processing, we take mean current (simplification)
        # In full version, would track per-batch membrane states
        mean_current = currents.mean(0)

        # LIF dynamics
        self.membrane = self.beta * self.membrane + mean_current

        # Spike generation (vectorized threshold comparison)
        spikes = (self.membrane >= self.threshold).float()

        # Soft reset
        self.membrane = self.membrane - spikes * self.threshold

        # Update spike history: shift and add new
        self.spike_history = torch.roll(self.spike_history, -1, dims=1)
        self.spike_history[:, -1] = spikes

        # Compute predictions if requested
        predictions = None
        if return_predictions:
            # Each neuron's rate
            rates = self.spike_history.mean(dim=1)  # [num_neurons]

            # Effective prediction weights
            eff_weights = (self.fast_pred_weights + self.slow_pred_weights) * self.neighbor_adj

            # Prediction: sigmoid(w_ij * rate_i + bias_ij)
            # For each neuron i, predict rate of each neighbor j
            # rates[:, None] broadcasts to [num_neurons, num_neurons]
            raw_pred = eff_weights * rates[:, None] + self.pred_bias
            predictions = torch.sigmoid(raw_pred) * self.neighbor_adj

        # Expand spikes to batch dimension
        spikes_batch = spikes.unsqueeze(0).expand(batch_size, -1)

        info = {
            'membrane': self.membrane.clone(),
            'rates': self.spike_history.mean(dim=1),
        }

        if squeeze:
            spikes_batch = spikes_batch.squeeze(0)

        return spikes_batch, predictions, info

    def compute_prediction_error(self) -> torch.Tensor:
        """
        Compute prediction error for all neuron pairs.

        Returns:
            error: [num_neurons, num_neurons] - actual minus predicted rates
        """
        # Actual neighbor rates
        actual_rates = self.spike_history.mean(dim=1)  # [num_neurons]

        # My rates for prediction
        my_rates = actual_rates

        # Predicted rates (what I predicted my neighbors would do)
        eff_weights = (self.fast_pred_weights + self.slow_pred_weights) * self.neighbor_adj
        raw_pred = eff_weights * my_rates[:, None] + self.pred_bias
        predicted_rates = torch.sigmoid(raw_pred)

        # Error: actual[j] - predicted[i,j] for neighbor j of neuron i
        # actual_rates[None, :] broadcasts to [num_neurons, num_neurons]
        error = (actual_rates[None, :] - predicted_rates) * self.neighbor_adj

        return error

    def compute_surprise(self) -> float:
        """Compute layer-wide surprise (for neuromodulation)."""
        error = self.compute_prediction_error()
        current_error = error.abs().sum() / (self.neighbor_adj.sum() + 1e-6)
        current_error = current_error.item()

        # Update history
        self.error_history[self.error_idx % 100] = current_error
        self.error_idx += 1

        # Surprise relative to baseline
        if self.error_idx < 10:
            return 1.0

        history = self.error_history[:min(self.error_idx, 100)]
        baseline = history.mean().item()
        std = history.std().item() + 1e-6

        surprise = (current_error - baseline) / std
        return float(torch.sigmoid(torch.tensor(surprise)).item())

    def local_update(self, surprise: float = 1.0):
        """
        Vectorized local plasticity update.

        All synapses updated in parallel.
        """
        error = self.compute_prediction_error()

        # Modulated learning rate
        effective_lr = self.fast_lr * surprise

        # My rates (for Hebbian term)
        my_rates = self.spike_history.mean(dim=1)

        # Delta: lr * error * pre_rate (Hebbian + error)
        # error is [num_neurons, num_neurons]
        # my_rates[:, None] broadcasts pre-synaptic rate
        delta = effective_lr * error * my_rates[:, None]

        with torch.no_grad():
            # Update fast weights (only for neighbors)
            self.fast_pred_weights.data += delta * self.neighbor_adj

            # Anti-Hebbian drift toward slow weights
            self.fast_pred_weights.data = (
                (1 - self.consolidation_rate) * self.fast_pred_weights.data +
                self.consolidation_rate * self.slow_pred_weights
            )

            # Update utility tracking
            significant_update = (error.abs() > 0.1).float()
            self.synapse_utility += significant_update * self.neighbor_adj
            self.synapse_age += self.neighbor_adj

    def consolidate(self):
        """Move stable fast patterns to slow weights."""
        with torch.no_grad():
            # Consolidate well-used synapses
            useful = self.synapse_utility > (self.synapse_age * 0.1)
            useful = useful & (self.neighbor_adj > 0)

            update = self.slow_lr * (self.fast_pred_weights - self.slow_pred_weights)
            self.slow_pred_weights[useful] += update[useful]

    def structural_plasticity(self, min_utility_ratio: float = 0.05) -> int:
        """Prune low-utility synapses."""
        with torch.no_grad():
            utility_ratio = self.synapse_utility / (self.synapse_age + 1)
            dead = (utility_ratio < min_utility_ratio) & (self.neighbor_adj > 0)

            num_dead = dead.sum().item()

            self.fast_pred_weights.data[dead] = 0
            self.slow_pred_weights[dead] = 0

            return int(num_dead)


class VectorizedPSANetwork(nn.Module):
    """
    Multi-layer vectorized PSA network.
    """

    def __init__(
        self,
        input_dim: int,
        layer_sizes: list = [256, 128, 64],
        prediction_windows: list = [5, 10, 20],
        neighbor_radius: int = 5,
        beta: float = 0.9,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.layer_sizes = layer_sizes

        # Create layers
        self.layers = nn.ModuleList()
        prev_dim = input_dim

        for size, window in zip(layer_sizes, prediction_windows):
            layer = VectorizedPSALayer(
                num_neurons=size,
                input_dim=prev_dim,
                neighbor_radius=neighbor_radius,
                beta=beta,
                prediction_window=window,
            )
            self.layers.append(layer)
            prev_dim = size

        # Global surprise tracking
        self.register_buffer('global_error_history', torch.zeros(100))
        self.error_idx = 0

    def reset(self):
        """Reset all layers."""
        for layer in self.layers:
            layer.reset()
        self.error_idx = 0

    def forward(
        self,
        x: torch.Tensor,
        return_all_layers: bool = False,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass through all layers.

        Args:
            x: Input [batch, input_dim] or [input_dim]
            return_all_layers: Whether to return intermediate spikes

        Returns:
            output: Top layer spikes
            info: Dict with layer info
        """
        info = {
            'layer_spikes': [],
            'layer_predictions': [],
            'surprise': 0.0,
        }

        current = x

        for layer in self.layers:
            spikes, predictions, layer_info = layer(current)
            info['layer_spikes'].append(spikes)
            info['layer_predictions'].append(predictions)
            current = spikes

        # Compute global surprise
        info['surprise'] = self._compute_global_surprise()

        return current, info

    def _compute_global_surprise(self) -> float:
        """Average surprise across layers."""
        surprises = [layer.compute_surprise() for layer in self.layers]
        return sum(surprises) / len(surprises)

    def local_update(self):
        """Update all layers."""
        global_surprise = self._compute_global_surprise()
        for layer in self.layers:
            layer.local_update(global_surprise)

    def consolidate(self):
        """Consolidate all layers."""
        for layer in self.layers:
            layer.consolidate()

    def predict_next_state(
        self,
        current_state: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, float]:
        """
        Predict next state given current state and action.

        This is the world model interface for planning/testing.

        Args:
            current_state: Current observation encoding
            action: Action to take

        Returns:
            predicted_next: Predicted next state encoding
            uncertainty: Prediction uncertainty
        """
        # Concatenate state and action
        state_action = torch.cat([current_state, action], dim=-1)

        # Forward through network
        output, info = self.forward(state_action)

        # Uncertainty from surprise
        uncertainty = info['surprise']

        return output, uncertainty
