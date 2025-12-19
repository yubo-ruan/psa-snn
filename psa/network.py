"""
PSA Network - Full predictive sparse assembly network.
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Optional
from .neuron import PSALayer
from .assembly import AssemblyDetector
from .synapse import DualWeightSynapse


class PSANetwork(nn.Module):
    """
    Multi-layer PSA network with:
    - Hierarchical predictive layers (different timescales)
    - Assembly detection at each layer
    - Global surprise modulation
    """

    def __init__(
        self,
        input_dim: int,
        layer_sizes: List[int] = [256, 128, 64],
        prediction_windows: List[int] = [5, 10, 20],  # Increasing timescales
        neighbor_radius: int = 5,
        beta: float = 0.9,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)

        # Create layers with increasing prediction windows (timescales)
        self.layers = nn.ModuleList()
        self.assembly_detectors = nn.ModuleList()

        prev_dim = input_dim
        for i, (size, window) in enumerate(zip(layer_sizes, prediction_windows)):
            layer = PSALayer(
                num_neurons=size,
                input_dim=prev_dim,
                neighbor_radius=neighbor_radius,
                beta=beta,
                prediction_window=window,
            )
            self.layers.append(layer)

            detector = AssemblyDetector(
                num_neurons=size,
                error_threshold=0.2,
                min_assembly_size=3,
            )
            self.assembly_detectors.append(detector)

            prev_dim = size

        # Inter-layer connections (top-down predictions)
        self.top_down = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.top_down.append(DualWeightSynapse(
                in_features=layer_sizes[i + 1],
                out_features=layer_sizes[i],
            ))

        # Global surprise tracking
        self.register_buffer('global_error_history', torch.zeros(100))
        self.error_idx = 0

    def reset(self):
        """Reset all layers."""
        for layer in self.layers:
            layer.reset()
        for detector in self.assembly_detectors:
            detector.spike_history.zero_()
            detector.coactivation_matrix.zero_()
        self.error_idx = 0

    def forward(
        self,
        x: torch.Tensor,
        return_assemblies: bool = False,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass through network.

        Args:
            x: Input [batch, input_dim] or [input_dim]
            return_assemblies: Whether to return assembly activations

        Returns:
            output: Top layer spikes
            info: Dictionary with intermediate results
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)

        info = {
            'layer_spikes': [],
            'layer_predictions': [],
            'surprise': 0.0,
        }

        current = x

        # Bottom-up pass
        for i, (layer, detector) in enumerate(zip(self.layers, self.assembly_detectors)):
            spikes, predictions = layer(current)
            info['layer_spikes'].append(spikes)
            info['layer_predictions'].append(predictions)

            # Update assembly detector
            if spikes.dim() == 1:
                detector.update_assemblies(spikes)
            else:
                detector.update_assemblies(spikes[0])  # Use first batch item

            current = spikes

        # Compute global surprise
        info['surprise'] = self._compute_global_surprise()

        if return_assemblies:
            info['assemblies'] = []
            for i, detector in enumerate(self.assembly_detectors):
                spikes = info['layer_spikes'][i]
                if spikes.dim() > 1:
                    spikes = spikes[0]
                info['assemblies'].append(detector.get_assembly_activations(spikes))

        return current, info

    def _compute_global_surprise(self) -> float:
        """Compute global surprise across all layers."""
        total_error = 0.0

        for layer in self.layers:
            total_error += layer.compute_surprise()

        avg_error = total_error / len(self.layers)

        # Update history
        self.global_error_history[self.error_idx % 100] = avg_error
        self.error_idx += 1

        # Surprise relative to history
        if self.error_idx < 10:
            return 0.5

        history = self.global_error_history[:min(self.error_idx, 100)]
        baseline = history.mean().item()
        std = history.std().item() + 1e-6

        surprise = (avg_error - baseline) / std
        return float(torch.sigmoid(torch.tensor(surprise)).item())

    def local_update(self):
        """Update all layers with local plasticity."""
        for layer in self.layers:
            layer.local_update()

    def consolidate(self):
        """Consolidate all layers."""
        for layer in self.layers:
            layer.consolidate()

    def structural_plasticity(self) -> int:
        """Run structural plasticity on all layers."""
        total_dead = 0
        for layer in self.layers:
            total_dead += layer.structural_plasticity()
        return total_dead

    def predict_next_input(self, current_input: torch.Tensor) -> torch.Tensor:
        """
        Predict what the next input will be.

        Uses top-down predictions from higher layers.

        Args:
            current_input: Current input [batch, input_dim] or [input_dim]

        Returns:
            predicted_next: Predicted next input
        """
        # Run forward to get layer activations
        _, info = self.forward(current_input)

        # Use top layer to predict bottom layer
        if len(self.top_down) > 0:
            top_spikes = info['layer_spikes'][-1]
            predicted = top_spikes

            for i in range(len(self.top_down) - 1, -1, -1):
                predicted = torch.sigmoid(self.top_down[i](predicted))

            return predicted

        return info['layer_spikes'][0]

    def prediction_uncertainty(self, x: torch.Tensor) -> float:
        """
        Estimate prediction uncertainty for input.

        High uncertainty = good target for exploration.
        """
        _, info = self.forward(x)

        # Uncertainty from prediction errors
        total_uncertainty = 0.0
        for layer in self.layers:
            for neuron in layer.neurons:
                error = neuron.compute_prediction_error()
                total_uncertainty += error.abs().mean().item()

        return total_uncertainty / len(self.layers)

    def get_assembly_summary(self) -> str:
        """Get summary of all assemblies in network."""
        lines = ["PSA Network Assembly Summary"]
        for i, detector in enumerate(self.assembly_detectors):
            lines.append(f"\nLayer {i}:")
            lines.append(detector.summary())
        return "\n".join(lines)
