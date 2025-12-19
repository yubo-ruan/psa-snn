"""
Assembly detection and tracking.

Assemblies are groups of neurons with low mutual prediction error.
They represent stable, bound concepts (objects, actions, etc.)
"""

import torch
import torch.nn as nn
from typing import List, Set, Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class Assembly:
    """A detected neural assembly."""
    id: int
    neuron_indices: Set[int]
    creation_time: int
    last_active: int
    activation_count: int
    mean_internal_error: float  # Low = good binding


class AssemblyDetector(nn.Module):
    """
    Detects and tracks neural assemblies based on prediction agreement.

    An assembly is a group of neurons that:
    1. Successfully predict each other (low mutual error)
    2. Fire together across time
    3. Are stable (persist over many timesteps)
    """

    def __init__(
        self,
        num_neurons: int,
        error_threshold: float = 0.2,  # Max error to be considered "good prediction"
        min_assembly_size: int = 3,
        max_assembly_size: int = 20,
        coactivation_window: int = 10,
        merge_overlap_threshold: float = 0.5,
    ):
        super().__init__()
        self.num_neurons = num_neurons
        self.error_threshold = error_threshold
        self.min_assembly_size = min_assembly_size
        self.max_assembly_size = max_assembly_size
        self.coactivation_window = coactivation_window
        self.merge_overlap_threshold = merge_overlap_threshold

        # Track pairwise prediction errors
        self.register_buffer(
            'error_matrix',
            torch.ones(num_neurons, num_neurons)  # High = bad prediction
        )

        # Track coactivation
        self.register_buffer(
            'coactivation_matrix',
            torch.zeros(num_neurons, num_neurons)
        )

        # Spike history for coactivation
        self.register_buffer(
            'spike_history',
            torch.zeros(num_neurons, coactivation_window)
        )

        # Detected assemblies
        self.assemblies: Dict[int, Assembly] = {}
        self.next_assembly_id = 0
        self.current_time = 0

    def update_error_matrix(
        self,
        neuron_idx: int,
        neighbor_indices: List[int],
        prediction_errors: torch.Tensor,
    ):
        """
        Update error matrix with new prediction errors.

        Uses exponential moving average for stability.
        """
        alpha = 0.1  # EMA smoothing factor

        for i, neighbor_idx in enumerate(neighbor_indices):
            error = prediction_errors[i].abs().item()

            # Symmetric update
            self.error_matrix[neuron_idx, neighbor_idx] = (
                (1 - alpha) * self.error_matrix[neuron_idx, neighbor_idx] +
                alpha * error
            )
            self.error_matrix[neighbor_idx, neuron_idx] = (
                (1 - alpha) * self.error_matrix[neighbor_idx, neuron_idx] +
                alpha * error
            )

    def update_coactivation(self, spikes: torch.Tensor):
        """
        Update coactivation matrix based on recent spikes.

        Args:
            spikes: Current spike pattern [num_neurons]
        """
        # Shift history
        self.spike_history = torch.roll(self.spike_history, -1, dims=1)
        self.spike_history[:, -1] = spikes

        # Compute coactivation over window
        # Two neurons coactivate if they both fire within the window
        active = self.spike_history.sum(dim=1) > 0  # [num_neurons]

        # Update coactivation matrix
        coactive = torch.outer(active.float(), active.float())
        alpha = 0.05
        self.coactivation_matrix = (
            (1 - alpha) * self.coactivation_matrix +
            alpha * coactive
        )

        self.current_time += 1

    def detect_assemblies(self) -> List[Assembly]:
        """
        Detect assemblies from current error and coactivation matrices.

        Uses a greedy algorithm:
        1. Find pairs with low mutual error AND high coactivation
        2. Grow clusters from these seed pairs
        3. Validate clusters as assemblies
        """
        # Compute affinity: low error AND high coactivation
        affinity = (1 - self.error_matrix) * self.coactivation_matrix

        # Threshold for "good" affinity
        threshold = 0.3

        # Find connected components in affinity graph
        adjacency = affinity > threshold

        # Greedy clustering
        visited = set()
        new_assemblies = []

        for seed in range(self.num_neurons):
            if seed in visited:
                continue

            # BFS to find cluster
            cluster = {seed}
            queue = [seed]

            while queue and len(cluster) < self.max_assembly_size:
                node = queue.pop(0)
                neighbors = torch.where(adjacency[node])[0].tolist()

                for neighbor in neighbors:
                    if neighbor not in cluster and neighbor not in visited:
                        cluster.add(neighbor)
                        queue.append(neighbor)

            visited.update(cluster)

            # Validate as assembly
            if len(cluster) >= self.min_assembly_size:
                # Compute mean internal error
                cluster_list = list(cluster)
                internal_errors = []
                for i in cluster_list:
                    for j in cluster_list:
                        if i != j:
                            internal_errors.append(
                                self.error_matrix[i, j].item()
                            )

                mean_error = np.mean(internal_errors) if internal_errors else 1.0

                if mean_error < self.error_threshold:
                    assembly = Assembly(
                        id=self.next_assembly_id,
                        neuron_indices=cluster,
                        creation_time=self.current_time,
                        last_active=self.current_time,
                        activation_count=1,
                        mean_internal_error=mean_error,
                    )
                    new_assemblies.append(assembly)
                    self.next_assembly_id += 1

        return new_assemblies

    def update_assemblies(self, spikes: torch.Tensor):
        """
        Update assembly tracking with current spike pattern.

        Args:
            spikes: Current spikes [num_neurons]
        """
        self.update_coactivation(spikes)

        # Check which existing assemblies are active
        for assembly_id, assembly in list(self.assemblies.items()):
            indices = list(assembly.neuron_indices)
            assembly_spikes = spikes[indices]

            # Assembly is active if majority of neurons fire
            if assembly_spikes.mean() > 0.3:
                assembly.last_active = self.current_time
                assembly.activation_count += 1

            # Remove stale assemblies
            if self.current_time - assembly.last_active > 1000:
                del self.assemblies[assembly_id]

        # Periodically detect new assemblies
        if self.current_time % 100 == 0:
            new_assemblies = self.detect_assemblies()

            for assembly in new_assemblies:
                # Check if it overlaps significantly with existing
                merged = False
                for existing_id, existing in self.assemblies.items():
                    overlap = len(
                        assembly.neuron_indices & existing.neuron_indices
                    ) / len(assembly.neuron_indices)

                    if overlap > self.merge_overlap_threshold:
                        # Merge into existing
                        existing.neuron_indices |= assembly.neuron_indices
                        existing.last_active = self.current_time
                        merged = True
                        break

                if not merged:
                    self.assemblies[assembly.id] = assembly

    def get_assembly_activations(self, spikes: torch.Tensor) -> Dict[int, float]:
        """
        Get activation level of each assembly.

        Args:
            spikes: Current spikes [num_neurons]

        Returns:
            Dictionary mapping assembly ID to activation level [0, 1]
        """
        activations = {}
        for assembly_id, assembly in self.assemblies.items():
            indices = list(assembly.neuron_indices)
            activations[assembly_id] = spikes[indices].mean().item()
        return activations

    def get_assembly_vector(self, spikes: torch.Tensor) -> torch.Tensor:
        """
        Get fixed-size assembly activation vector.

        Useful for downstream readout.

        Args:
            spikes: Current spikes [num_neurons]

        Returns:
            Assembly activations [num_assemblies]
        """
        if not self.assemblies:
            return torch.zeros(1, device=spikes.device)

        activations = []
        for assembly_id in sorted(self.assemblies.keys()):
            indices = list(self.assemblies[assembly_id].neuron_indices)
            activations.append(spikes[indices].mean())

        return torch.stack(activations)

    def summary(self) -> str:
        """Get summary of detected assemblies."""
        lines = [f"AssemblyDetector: {len(self.assemblies)} assemblies"]
        for assembly_id, assembly in sorted(self.assemblies.items()):
            lines.append(
                f"  Assembly {assembly_id}: "
                f"{len(assembly.neuron_indices)} neurons, "
                f"error={assembly.mean_internal_error:.3f}, "
                f"activations={assembly.activation_count}"
            )
        return "\n".join(lines)
