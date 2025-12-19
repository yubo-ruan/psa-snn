"""
PSA Neurons with predictive capabilities.

Each neuron:
1. Integrates input (LIF dynamics)
2. Predicts neighbor firing rates in a window
3. Updates based on prediction error (local learning)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple
import math


class PSANeuron(nn.Module):
    """
    Single PSA neuron with:
    - LIF dynamics
    - Neighbor spike rate prediction
    - Local plasticity based on prediction error
    """

    def __init__(
        self,
        neuron_id: int,
        num_neighbors: int,
        beta: float = 0.9,  # Membrane decay
        threshold: float = 1.0,
        prediction_window: int = 5,  # Predict over this many timesteps
        fast_lr: float = 0.01,
        slow_lr: float = 0.001,
        consolidation_rate: float = 0.001,
    ):
        super().__init__()
        self.neuron_id = neuron_id
        self.num_neighbors = num_neighbors
        self.beta = beta
        self.threshold = threshold
        self.prediction_window = prediction_window
        self.fast_lr = fast_lr
        self.slow_lr = slow_lr
        self.consolidation_rate = consolidation_rate

        # Membrane potential
        self.register_buffer('membrane', torch.zeros(1))

        # Prediction weights: predict each neighbor's firing rate
        # Fast weights (episodic, high plasticity)
        self.fast_weights = nn.Parameter(torch.randn(num_neighbors) * 0.1)
        # Slow weights (consolidated, low plasticity)
        self.register_buffer('slow_weights', torch.zeros(num_neighbors))

        # Prediction bias
        self.pred_bias = nn.Parameter(torch.zeros(num_neighbors))

        # Track recent activity for rate estimation
        self.register_buffer('spike_history', torch.zeros(prediction_window))
        self.register_buffer('neighbor_history', torch.zeros(num_neighbors, prediction_window))

        # Utility tracking for structural plasticity
        self.register_buffer('synapse_utility', torch.zeros(num_neighbors))
        self.register_buffer('synapse_age', torch.zeros(num_neighbors))

    def reset(self):
        """Reset membrane and history."""
        self.membrane.zero_()
        self.spike_history.zero_()
        self.neighbor_history.zero_()

    def forward(
        self,
        input_current: torch.Tensor,
        neighbor_spikes: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            input_current: Input current to this neuron [1]
            neighbor_spikes: Spikes from neighbors [num_neighbors]

        Returns:
            spike: Whether this neuron spiked [1]
            prediction: Predicted neighbor rates [num_neighbors]
        """
        # LIF dynamics
        self.membrane = self.beta * self.membrane + input_current

        # Spike generation (hard threshold)
        spike = (self.membrane >= self.threshold).float()

        # Soft reset
        self.membrane = self.membrane - spike * self.threshold

        # Update spike history (shift and add new spike)
        self.spike_history = torch.roll(self.spike_history, -1)
        self.spike_history[-1] = spike

        # Update neighbor history if provided
        if neighbor_spikes is not None:
            self.neighbor_history = torch.roll(self.neighbor_history, -1, dims=1)
            self.neighbor_history[:, -1] = neighbor_spikes

        # Predict neighbor firing rates based on my recent activity
        # Combined fast + slow weights
        effective_weights = self.fast_weights + self.slow_weights
        my_rate = self.spike_history.mean()
        prediction = torch.sigmoid(effective_weights * my_rate + self.pred_bias)

        return spike, prediction

    def compute_prediction_error(self) -> torch.Tensor:
        """Compute prediction error (actual - predicted neighbor rates)."""
        actual_rates = self.neighbor_history.mean(dim=1)
        effective_weights = self.fast_weights + self.slow_weights
        my_rate = self.spike_history.mean()
        predicted_rates = torch.sigmoid(effective_weights * my_rate + self.pred_bias)

        return actual_rates - predicted_rates

    def local_update(self, surprise: float = 1.0):
        """
        Local plasticity update based on prediction error.

        Args:
            surprise: Global surprise signal (modulates learning rate)
        """
        error = self.compute_prediction_error()

        # Modulated learning rate
        effective_lr = self.fast_lr * surprise

        # Update fast weights (Hebbian + error)
        my_rate = self.spike_history.mean()
        delta = effective_lr * error * my_rate

        with torch.no_grad():
            self.fast_weights.data += delta

            # Anti-Hebbian drift: decay fast weights toward slow weights
            self.fast_weights.data = (
                (1 - self.consolidation_rate) * self.fast_weights.data +
                self.consolidation_rate * self.slow_weights
            )

            # Update utility (how often this synapse had non-trivial error)
            self.synapse_utility += (error.abs() > 0.1).float()
            self.synapse_age += 1

    def consolidate(self):
        """Move stable fast weight patterns to slow weights."""
        with torch.no_grad():
            # Only consolidate well-used synapses
            useful = self.synapse_utility > (self.synapse_age * 0.1)

            # Move fast → slow for useful synapses
            self.slow_weights[useful] += self.slow_lr * (
                self.fast_weights[useful] - self.slow_weights[useful]
            )

    def structural_plasticity(self, min_utility_ratio: float = 0.05):
        """Kill useless synapses, potentially spawn new ones."""
        with torch.no_grad():
            # Compute utility ratio
            utility_ratio = self.synapse_utility / (self.synapse_age + 1)

            # Kill synapses with very low utility
            dead = utility_ratio < min_utility_ratio
            self.fast_weights.data[dead] = 0
            self.slow_weights[dead] = 0

            return dead.sum().item()


class PSALayer(nn.Module):
    """
    Layer of PSA neurons with lateral connections.

    Each neuron predicts its neighbors' activity.
    """

    def __init__(
        self,
        num_neurons: int,
        input_dim: int,
        neighbor_radius: int = 5,  # Each neuron connects to neighbors within this radius
        beta: float = 0.9,
        threshold: float = 1.0,
        prediction_window: int = 5,
    ):
        super().__init__()
        self.num_neurons = num_neurons
        self.input_dim = input_dim
        self.neighbor_radius = neighbor_radius

        # Input weights (feedforward)
        self.input_weights = nn.Parameter(torch.randn(num_neurons, input_dim) * 0.1)

        # Create neurons with their neighbor connections
        self.neurons = nn.ModuleList()
        for i in range(num_neurons):
            # Neighbors are within radius (circular)
            num_neighbors = min(2 * neighbor_radius, num_neurons - 1)
            self.neurons.append(PSANeuron(
                neuron_id=i,
                num_neighbors=num_neighbors,
                beta=beta,
                threshold=threshold,
                prediction_window=prediction_window,
            ))

        # Precompute neighbor indices for each neuron
        self.neighbor_indices = []
        for i in range(num_neurons):
            neighbors = []
            for offset in range(-neighbor_radius, neighbor_radius + 1):
                if offset != 0:
                    j = (i + offset) % num_neurons
                    neighbors.append(j)
            self.neighbor_indices.append(neighbors[:2 * neighbor_radius])

        # Layer-wide surprise tracking
        self.register_buffer('error_history', torch.zeros(100))
        self.error_idx = 0

    def reset(self):
        """Reset all neurons."""
        for neuron in self.neurons:
            neuron.reset()
        self.error_history.zero_()
        self.error_idx = 0

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through layer.

        Args:
            x: Input [batch, input_dim] or [input_dim]

        Returns:
            spikes: Output spikes [batch, num_neurons] or [num_neurons]
            predictions: Predicted neighbor rates [num_neurons, num_neighbors]
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        batch_size = x.shape[0]

        # Compute input currents for all neurons
        currents = F.linear(x, self.input_weights)  # [batch, num_neurons]

        # Process each neuron (can be parallelized in practice)
        spikes = torch.zeros(batch_size, self.num_neurons, device=x.device)
        predictions = torch.zeros(self.num_neurons, 2 * self.neighbor_radius, device=x.device)

        # Get previous spikes for neighbor input
        prev_spikes = torch.stack([n.spike_history[-1] for n in self.neurons])

        for i, neuron in enumerate(self.neurons):
            # Get neighbor spikes
            neighbor_idx = self.neighbor_indices[i]
            neighbor_spikes = prev_spikes[neighbor_idx]

            # Forward pass (for each batch item)
            for b in range(batch_size):
                spike, pred = neuron(currents[b, i:i+1], neighbor_spikes)
                spikes[b, i] = spike

            predictions[i] = pred

        return spikes.squeeze(0), predictions

    def compute_surprise(self) -> float:
        """Compute layer-wide surprise (for neuromodulation)."""
        # Collect prediction errors from all neurons
        errors = []
        for neuron in self.neurons:
            errors.append(neuron.compute_prediction_error().abs().mean())

        current_error = torch.stack(errors).mean().item()

        # Update error history
        self.error_history[self.error_idx % 100] = current_error
        self.error_idx += 1

        # Surprise = current error relative to history
        if self.error_idx < 10:
            return 1.0  # Not enough history

        baseline = self.error_history[:min(self.error_idx, 100)].mean().item()
        std = self.error_history[:min(self.error_idx, 100)].std().item() + 1e-6

        surprise = (current_error - baseline) / std
        return float(torch.sigmoid(torch.tensor(surprise)).item())

    def local_update(self):
        """Update all neurons with surprise-gated plasticity."""
        surprise = self.compute_surprise()
        for neuron in self.neurons:
            neuron.local_update(surprise)

    def consolidate(self):
        """Consolidate all neurons."""
        for neuron in self.neurons:
            neuron.consolidate()

    def structural_plasticity(self) -> int:
        """Run structural plasticity on all neurons."""
        total_dead = 0
        for neuron in self.neurons:
            total_dead += neuron.structural_plasticity()
        return total_dead
