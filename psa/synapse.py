"""
PSA Synapses with dual weight systems and structural plasticity.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class SynapseStats:
    """Statistics for a synapse."""
    weight: float
    age: int
    utility: float
    utility_ratio: float
    is_alive: bool


class Synapse:
    """
    Single synapse with:
    - Fast weight (episodic, high plasticity)
    - Slow weight (consolidated, low plasticity)
    - Utility tracking for structural plasticity
    """

    def __init__(
        self,
        pre_id: int,
        post_id: int,
        init_weight: float = 0.1,
        fast_lr: float = 0.01,
        slow_lr: float = 0.001,
        decay_rate: float = 0.9999,
    ):
        self.pre_id = pre_id
        self.post_id = post_id

        # Dual weights
        self.fast_weight = init_weight
        self.slow_weight = 0.0

        # Learning rates
        self.fast_lr = fast_lr
        self.slow_lr = slow_lr
        self.decay_rate = decay_rate

        # Utility tracking
        self.age = 0
        self.utility = 0.0
        self.recent_updates = []

        # Alive flag
        self.alive = True

    @property
    def weight(self) -> float:
        """Effective weight = fast + slow."""
        return self.fast_weight + self.slow_weight

    def update(
        self,
        pre_spike: bool,
        post_spike: bool,
        prediction_error: float,
        surprise: float,
    ):
        """
        Update synapse based on activity and prediction error.

        Args:
            pre_spike: Did presynaptic neuron fire?
            post_spike: Did postsynaptic neuron fire?
            prediction_error: Error in predicting post from pre
            surprise: Global surprise signal
        """
        if not self.alive:
            return

        self.age += 1

        # Only update if presynaptic neuron fired
        if pre_spike:
            # Modulated learning rate
            effective_lr = self.fast_lr * surprise

            # Update based on prediction error (not just Hebbian)
            delta = effective_lr * prediction_error

            self.fast_weight += delta

            # Track utility (did this synapse matter?)
            if abs(prediction_error) > 0.1:
                self.utility += 1

        # Anti-Hebbian drift
        self.fast_weight *= self.decay_rate

        # Track recent updates for consolidation
        self.recent_updates.append(self.fast_weight)
        if len(self.recent_updates) > 100:
            self.recent_updates.pop(0)

    def consolidate(self):
        """Move stable fast patterns to slow weight."""
        if not self.alive or len(self.recent_updates) < 50:
            return

        # Check if fast weight is stable
        recent = self.recent_updates[-50:]
        variance = torch.tensor(recent).var().item()

        if variance < 0.01:  # Stable
            # Move toward slow weight
            self.slow_weight += self.slow_lr * (self.fast_weight - self.slow_weight)

    def should_die(self, min_utility_ratio: float = 0.05) -> bool:
        """Check if this synapse should be pruned."""
        if self.age < 100:  # Give it time to prove useful
            return False

        utility_ratio = self.utility / self.age
        return utility_ratio < min_utility_ratio

    def die(self):
        """Mark synapse as dead."""
        self.alive = False
        self.fast_weight = 0.0
        self.slow_weight = 0.0

    def stats(self) -> SynapseStats:
        """Get synapse statistics."""
        return SynapseStats(
            weight=self.weight,
            age=self.age,
            utility=self.utility,
            utility_ratio=self.utility / max(self.age, 1),
            is_alive=self.alive,
        )


class DualWeightSynapse(nn.Module):
    """
    Vectorized dual-weight synapse for efficient GPU computation.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        fast_lr: float = 0.01,
        slow_lr: float = 0.001,
        decay_rate: float = 0.9999,
        consolidation_rate: float = 0.001,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fast_lr = fast_lr
        self.slow_lr = slow_lr
        self.decay_rate = decay_rate
        self.consolidation_rate = consolidation_rate

        # Fast weights (learnable)
        self.fast_weights = nn.Parameter(
            torch.randn(out_features, in_features) * 0.1
        )

        # Slow weights (buffer, not directly optimized)
        self.register_buffer(
            'slow_weights',
            torch.zeros(out_features, in_features)
        )

        # Utility and age tracking
        self.register_buffer(
            'utility',
            torch.zeros(out_features, in_features)
        )
        self.register_buffer(
            'age',
            torch.zeros(out_features, in_features)
        )

        # Alive mask
        self.register_buffer(
            'alive_mask',
            torch.ones(out_features, in_features, dtype=torch.bool)
        )

    @property
    def weight(self) -> torch.Tensor:
        """Effective weight = fast + slow, masked by alive."""
        return (self.fast_weights + self.slow_weights) * self.alive_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using effective weights."""
        return torch.matmul(x, self.weight.t())

    def local_update(
        self,
        pre_activity: torch.Tensor,
        post_activity: torch.Tensor,
        prediction_error: torch.Tensor,
        surprise: float,
    ):
        """
        Local plasticity update.

        Args:
            pre_activity: Presynaptic activity [in_features] or [batch, in_features]
            post_activity: Postsynaptic activity [out_features] or [batch, out_features]
            prediction_error: Error for each output [out_features] or [batch, out_features]
            surprise: Global surprise signal
        """
        if pre_activity.dim() == 1:
            pre_activity = pre_activity.unsqueeze(0)
        if post_activity.dim() == 1:
            post_activity = post_activity.unsqueeze(0)
        if prediction_error.dim() == 1:
            prediction_error = prediction_error.unsqueeze(0)

        # Average over batch
        pre_activity = pre_activity.mean(0)
        prediction_error = prediction_error.mean(0)

        # Modulated learning rate
        effective_lr = self.fast_lr * surprise

        with torch.no_grad():
            # Outer product: which pre-post pairs should update
            # Error-modulated Hebbian: Δw_ij = lr * error_j * pre_i
            delta = effective_lr * torch.outer(prediction_error, pre_activity)

            # Apply update only to alive synapses
            self.fast_weights.data += delta * self.alive_mask

            # Anti-Hebbian drift
            self.fast_weights.data *= self.decay_rate

            # Update utility (synapses with large updates are useful)
            self.utility += (delta.abs() > 0.001).float()
            self.age += self.alive_mask.float()

    def consolidate(self):
        """Move stable fast patterns to slow weights."""
        with torch.no_grad():
            # Consolidate based on utility ratio
            useful = self.utility > (self.age * 0.1)

            self.slow_weights[useful] += self.slow_lr * (
                self.fast_weights[useful] - self.slow_weights[useful]
            )

    def structural_plasticity(self, min_utility_ratio: float = 0.05) -> int:
        """Prune useless synapses."""
        with torch.no_grad():
            # Only consider mature synapses
            mature = self.age > 100

            # Compute utility ratio
            utility_ratio = self.utility / (self.age + 1)

            # Kill low-utility mature synapses
            should_die = mature & (utility_ratio < min_utility_ratio)

            num_dead = should_die.sum().item()

            self.alive_mask[should_die] = False
            self.fast_weights.data[should_die] = 0
            self.slow_weights[should_die] = 0

            return num_dead

    def spawn_synapses(
        self,
        active_pre: torch.Tensor,
        active_post: torch.Tensor,
        spawn_rate: float = 0.01,
    ) -> int:
        """
        Spawn new synapses between active neurons.

        Args:
            active_pre: Which presynaptic neurons are active [in_features]
            active_post: Which postsynaptic neurons are active [out_features]
            spawn_rate: Probability of spawning per dead synapse
        """
        with torch.no_grad():
            # Find dead synapses
            dead = ~self.alive_mask

            # Candidate positions: dead synapses between active neurons
            candidates = dead & torch.outer(active_post > 0, active_pre > 0)

            # Random spawn
            spawn_mask = torch.rand_like(candidates.float()) < spawn_rate
            spawn = candidates & spawn_mask

            num_spawned = spawn.sum().item()

            # Revive spawned synapses
            self.alive_mask[spawn] = True
            self.fast_weights.data[spawn] = torch.randn(num_spawned, device=self.fast_weights.device) * 0.1
            self.slow_weights[spawn] = 0
            self.utility[spawn] = 0
            self.age[spawn] = 0

            return num_spawned
