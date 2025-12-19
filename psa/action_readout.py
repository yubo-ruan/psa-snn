"""
3-Factor Learning Rule for Action Readout.

The action readout layer learns to map PSA representations to actions
using a biologically-plausible 3-factor rule:

    Δw_ij = η * pre_i * post_j * (target_j - output_j)

Where:
- pre_i: pre-synaptic activity (PSA spike rate)
- post_j: post-synaptic activity (action neuron rate)
- (target - output): teaching signal (from demonstration)

This is local (per-synapse), requires only local information, and
has biological support (dopamine as teaching signal).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple


class ThreeFactorReadout(nn.Module):
    """
    Action readout layer with 3-factor learning rule.

    Maps PSA spike patterns to continuous actions.
    """

    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        lr: float = 0.01,  # Moderate learning rate
        eligibility_decay: float = 0.5,
        noise_std: float = 0.05,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.eligibility_decay = eligibility_decay
        self.noise_std = noise_std

        # Two-layer readout: input → hidden → action
        self.w1 = nn.Parameter(torch.randn(hidden_dim, input_dim) * 0.1)
        self.b1 = nn.Parameter(torch.zeros(hidden_dim))
        self.w2 = nn.Parameter(torch.randn(action_dim, hidden_dim) * 0.1)
        self.b2 = nn.Parameter(torch.zeros(action_dim))

        # Eligibility traces (for temporal credit assignment)
        self.register_buffer('e1', torch.zeros(hidden_dim, input_dim))
        self.register_buffer('e2', torch.zeros(action_dim, hidden_dim))

        # Activity traces
        self.register_buffer('pre_activity', torch.zeros(input_dim))
        self.register_buffer('hidden_activity', torch.zeros(hidden_dim))
        self.register_buffer('output_activity', torch.zeros(action_dim))

    def reset(self):
        """Reset eligibility traces."""
        self.e1.zero_()
        self.e2.zero_()
        self.pre_activity.zero_()
        self.hidden_activity.zero_()
        self.output_activity.zero_()

    def forward(
        self,
        x: torch.Tensor,
        add_noise: bool = True,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass.

        Args:
            x: Input from PSA [input_dim] or [batch, input_dim]
            add_noise: Whether to add exploration noise

        Returns:
            action: Output action [action_dim]
            info: Dict with activities
        """
        squeeze = x.dim() == 1
        if squeeze:
            x = x.unsqueeze(0)

        # Store pre-synaptic activity
        self.pre_activity = x.mean(0).detach()

        # Hidden layer (ReLU activation)
        hidden = F.relu(F.linear(x, self.w1, self.b1))
        self.hidden_activity = hidden.mean(0).detach()

        # Output layer (tanh for bounded actions)
        output = torch.tanh(F.linear(hidden, self.w2, self.b2))
        self.output_activity = output.mean(0).detach()

        # Add exploration noise during training
        if add_noise and self.training:
            noise = torch.randn_like(output) * self.noise_std
            output = torch.clamp(output + noise, -1, 1)

        # Update eligibility traces
        self._update_eligibility()

        info = {
            'pre_activity': self.pre_activity,
            'hidden_activity': self.hidden_activity,
            'output_activity': self.output_activity,
        }

        if squeeze:
            output = output.squeeze(0)

        return output, info

    def _update_eligibility(self):
        """Update eligibility traces based on current activity."""
        # Eligibility: pre * post (Hebbian part of 3-factor rule)
        # Layer 1: pre_activity × hidden_activity
        e1_new = torch.outer(self.hidden_activity, self.pre_activity)
        self.e1 = self.eligibility_decay * self.e1 + (1 - self.eligibility_decay) * e1_new

        # Layer 2: hidden_activity × output_activity
        e2_new = torch.outer(self.output_activity, self.hidden_activity)
        self.e2 = self.eligibility_decay * self.e2 + (1 - self.eligibility_decay) * e2_new

    def local_update(self, target_action: torch.Tensor):
        """
        Apply 3-factor learning rule.

        Simplified version: Δw = lr * pre * error (delta rule)

        Args:
            target_action: Demonstration action [action_dim]
        """
        if target_action.dim() > 1:
            target_action = target_action.squeeze(0)

        # Teaching signal (error)
        error = target_action - self.output_activity
        error = torch.clamp(error, -1, 1)

        with torch.no_grad():
            # Layer 2 update: Δw2 = lr * error ⊗ hidden (delta rule)
            delta2 = self.lr * torch.outer(error, self.hidden_activity)
            delta2 = torch.clamp(delta2, -0.1, 0.1)
            self.w2.data += delta2
            self.b2.data += self.lr * error * 0.1

            # Layer 1 update: backprop error through layer 2
            error_hidden = F.linear(error, self.w2.t())
            error_hidden = error_hidden * (self.hidden_activity > 0).float()  # ReLU derivative
            error_hidden = torch.clamp(error_hidden, -1, 1)

            delta1 = self.lr * torch.outer(error_hidden, self.pre_activity)
            delta1 = torch.clamp(delta1, -0.1, 0.1)
            self.w1.data += delta1
            self.b1.data += self.lr * error_hidden * 0.1

            # Weight decay
            self.w1.data *= 0.9999
            self.w2.data *= 0.9999

            # Clamp weights
            self.w1.data.clamp_(-3, 3)
            self.w2.data.clamp_(-3, 3)


class ThreeFactorWithEligibility(nn.Module):
    """
    3-Factor readout with proper eligibility traces for temporal credit assignment.

    The key insight: actions taken now affect outcomes later, so we need to
    remember which synapses were active when the action was taken.

    Eligibility trace tracks: e_ij(t) = decay * e_ij(t-1) + pre_i(t) * post_j(t)
    Weight update uses: Δw_ij = lr * error(t) * e_ij(t)

    This allows delayed rewards/errors to update synapses that were active
    in the past, solving temporal credit assignment locally.
    """

    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        lr: float = 0.01,
        eligibility_decay: float = 0.7,  # Higher = longer memory
        noise_std: float = 0.05,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.eligibility_decay = eligibility_decay
        self.noise_std = noise_std

        # Two-layer readout
        self.w1 = nn.Parameter(torch.randn(hidden_dim, input_dim) * 0.1)
        self.b1 = nn.Parameter(torch.zeros(hidden_dim))
        self.w2 = nn.Parameter(torch.randn(action_dim, hidden_dim) * 0.1)
        self.b2 = nn.Parameter(torch.zeros(action_dim))

        # Eligibility traces - same shape as weights
        self.register_buffer('e1', torch.zeros(hidden_dim, input_dim))
        self.register_buffer('e2', torch.zeros(action_dim, hidden_dim))

        # Activity traces for computing eligibility
        self.register_buffer('pre_activity', torch.zeros(input_dim))
        self.register_buffer('hidden_activity', torch.zeros(hidden_dim))
        self.register_buffer('output_activity', torch.zeros(action_dim))

        # Track recent errors for adaptive learning
        self.register_buffer('error_history', torch.zeros(20))
        self.error_idx = 0

    def reset(self):
        """Reset eligibility traces for new episode."""
        self.e1.zero_()
        self.e2.zero_()
        self.pre_activity.zero_()
        self.hidden_activity.zero_()
        self.output_activity.zero_()
        self.error_history.zero_()
        self.error_idx = 0

    def forward(
        self,
        x: torch.Tensor,
        add_noise: bool = True,
    ) -> Tuple[torch.Tensor, Dict]:
        """Forward pass - also updates eligibility traces."""
        if x.dim() == 2:
            x = x.squeeze(0)

        # Store pre-synaptic activity
        self.pre_activity = x.detach()

        # Hidden layer
        hidden = F.relu(F.linear(x, self.w1, self.b1))
        self.hidden_activity = hidden.detach()

        # Output layer
        output = torch.tanh(F.linear(hidden, self.w2, self.b2))
        self.output_activity = output.detach()

        # Add exploration noise during training
        if add_noise and self.training:
            noise = torch.randn_like(output) * self.noise_std
            output = torch.clamp(output + noise, -1, 1)

        # Update eligibility traces (accumulate Hebbian product)
        self._update_eligibility()

        info = {
            'pre_activity': self.pre_activity,
            'hidden_activity': self.hidden_activity,
            'output_activity': self.output_activity,
            'e1_norm': self.e1.norm().item(),
            'e2_norm': self.e2.norm().item(),
        }

        return output, info

    def _update_eligibility(self):
        """
        Update eligibility traces.

        e(t) = decay * e(t-1) + pre(t) * post(t)

        This accumulates a memory of recent synaptic coincidences.
        """
        # Layer 1: pre_activity → hidden_activity
        e1_new = torch.outer(self.hidden_activity, self.pre_activity)
        self.e1 = self.eligibility_decay * self.e1 + e1_new

        # Layer 2: hidden_activity → output_activity
        e2_new = torch.outer(self.output_activity, self.hidden_activity)
        self.e2 = self.eligibility_decay * self.e2 + e2_new

        # Clamp to prevent explosion
        self.e1.clamp_(-5, 5)
        self.e2.clamp_(-5, 5)

    def local_update(self, target_action: torch.Tensor):
        """
        Apply 3-factor learning rule using eligibility traces.

        Δw = lr * error * eligibility

        The eligibility trace tells us which synapses contributed to recent
        actions, so the error signal can update them appropriately even if
        the error arrives with some delay.

        Args:
            target_action: Demonstration action [action_dim]
        """
        if target_action.dim() > 1:
            target_action = target_action.squeeze(0)

        # Compute error (teaching signal)
        error = target_action.float() - self.output_activity
        error = torch.clamp(error, -1, 1)

        # Track error for adaptive learning
        self._update_error_history(error)

        with torch.no_grad():
            # Layer 2 update: Δw2 = lr * error * e2
            # e2 already has shape [action_dim, hidden_dim]
            # error has shape [action_dim]
            # We want: delta2[i,j] = lr * error[i] * e2[i,j]
            delta2 = self.lr * error.unsqueeze(1) * self.e2
            delta2 = torch.clamp(delta2, -0.1, 0.1)
            self.w2.data += delta2
            self.b2.data += self.lr * error * 0.1

            # Layer 1 update: backprop error through layer 2, use e1
            error_hidden = F.linear(error, self.w2.t())
            error_hidden = error_hidden * (self.hidden_activity > 0).float()
            error_hidden = torch.clamp(error_hidden, -1, 1)

            # delta1[i,j] = lr * error_hidden[i] * e1[i,j]
            delta1 = self.lr * error_hidden.unsqueeze(1) * self.e1
            delta1 = torch.clamp(delta1, -0.1, 0.1)
            self.w1.data += delta1
            self.b1.data += self.lr * error_hidden * 0.1

            # Weight decay
            self.w1.data *= 0.9999
            self.w2.data *= 0.9999

            # Clamp weights
            self.w1.data.clamp_(-3, 3)
            self.w2.data.clamp_(-3, 3)

    def _update_error_history(self, error: torch.Tensor):
        """Track recent errors for monitoring."""
        error_mag = error.abs().mean().item()
        self.error_history[self.error_idx % 20] = error_mag
        self.error_idx += 1


class PSAWithEligibilityReadout(nn.Module):
    """
    PSA system with eligibility-trace-based action readout.

    Uses ThreeFactorWithEligibility for temporal credit assignment.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        psa_neurons: int = 64,
        readout_hidden: int = 32,
        eligibility_decay: float = 0.7,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        from psa.predictive_psa import PredictivePSALayer

        self.psa = PredictivePSALayer(
            obs_dim=obs_dim,
            action_dim=action_dim,
            num_neurons=psa_neurons,
        )

        self.readout = ThreeFactorWithEligibility(
            input_dim=psa_neurons,
            action_dim=action_dim,
            hidden_dim=readout_hidden,
            eligibility_decay=eligibility_decay,
        )

        self.register_buffer('last_action', torch.zeros(action_dim))

    def reset(self):
        """Reset all state."""
        self.psa.reset()
        self.readout.reset()
        self.last_action.zero_()

    def forward(
        self,
        obs: torch.Tensor,
        teacher_action: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """Forward pass."""
        spikes, prediction, psa_info = self.psa(obs, self.last_action)
        action, readout_info = self.readout(spikes, add_noise=self.training)
        self.last_action = action.detach().clone()

        info = {
            'psa_info': psa_info,
            'readout_info': readout_info,
            'spikes': spikes,
        }

        return action, prediction, info

    def update(
        self,
        next_obs: torch.Tensor,
        demo_action: Optional[torch.Tensor] = None,
    ):
        """Local learning update."""
        self.psa.local_update(next_obs)
        if demo_action is not None:
            self.readout.local_update(demo_action)

    def consolidate(self):
        """Consolidate learned patterns."""
        self.psa.consolidate()


class PSAWithActionReadout(nn.Module):
    """
    Complete PSA system with action readout.

    Combines:
    1. Predictive PSA (world model with local learning)
    2. 3-factor action readout (imitation with local learning)
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        psa_neurons: int = 64,
        readout_hidden: int = 32,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # Import here to avoid circular dependency
        from psa.predictive_psa import PredictivePSALayer

        # Single PSA layer for simplicity
        self.psa = PredictivePSALayer(
            obs_dim=obs_dim,
            action_dim=action_dim,
            num_neurons=psa_neurons,
        )

        # Action readout
        self.readout = ThreeFactorReadout(
            input_dim=psa_neurons,
            action_dim=action_dim,
            hidden_dim=readout_hidden,
        )

        # Store last action for conditioning
        self.register_buffer('last_action', torch.zeros(action_dim))

    def reset(self):
        """Reset all state."""
        self.psa.reset()
        self.readout.reset()
        self.last_action.zero_()

    def forward(
        self,
        obs: torch.Tensor,
        teacher_action: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Forward pass.

        Args:
            obs: Current observation [obs_dim]
            teacher_action: Demo action for training (optional)

        Returns:
            action: Predicted action [action_dim]
            prediction: Predicted next observation [obs_dim]
            info: Debug info
        """
        # PSA forward (conditioned on last action)
        spikes, prediction, psa_info = self.psa(obs, self.last_action)

        # Action readout from PSA spikes
        action, readout_info = self.readout(spikes, add_noise=self.training)

        # Store action for next step
        self.last_action = action.detach().clone()

        info = {
            'psa_info': psa_info,
            'readout_info': readout_info,
            'spikes': spikes,
        }

        return action, prediction, info

    def update(
        self,
        next_obs: torch.Tensor,
        demo_action: Optional[torch.Tensor] = None,
    ):
        """
        Local learning update.

        Args:
            next_obs: Actual next observation (for world model)
            demo_action: Demonstration action (for imitation)
        """
        # Update PSA world model
        self.psa.local_update(next_obs)

        # Update action readout if demo available
        if demo_action is not None:
            self.readout.local_update(demo_action)

    def consolidate(self):
        """Consolidate learned patterns."""
        self.psa.consolidate()


class ChunkedActionReadout(nn.Module):
    """
    Action readout that outputs K-step action chunks.

    Benefits for manipulation:
    - Smoother control (less jerk)
    - Better grasp stability
    - More precise placement

    Training: 3-factor rule applied to entire chunk, weighted toward earlier steps.
    Execution: Receding horizon (execute first action, replan every step).
    """

    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        chunk_size: int = 5,
        hidden_dim: int = 64,
        lr: float = 0.01,
        noise_std: float = 0.03,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.noise_std = noise_std

        # Output dimension is chunk_size * action_dim
        self.output_dim = chunk_size * action_dim

        # Two-layer readout: input → hidden → action_chunk
        self.w1 = nn.Parameter(torch.randn(hidden_dim, input_dim) * 0.1)
        self.b1 = nn.Parameter(torch.zeros(hidden_dim))
        self.w2 = nn.Parameter(torch.randn(self.output_dim, hidden_dim) * 0.1)
        self.b2 = nn.Parameter(torch.zeros(self.output_dim))

        # Activity traces
        self.register_buffer('pre_activity', torch.zeros(input_dim))
        self.register_buffer('hidden_activity', torch.zeros(hidden_dim))
        self.register_buffer('output_activity', torch.zeros(self.output_dim))

        # Temporal weighting for chunk error (earlier steps weighted more)
        # Exponential decay: [1.0, 0.8, 0.64, 0.51, 0.41] for K=5
        weights = torch.tensor([0.8 ** i for i in range(chunk_size)])
        weights = weights / weights.sum()  # Normalize
        self.register_buffer('temporal_weights', weights)

    def reset(self):
        """Reset activity traces."""
        self.pre_activity.zero_()
        self.hidden_activity.zero_()
        self.output_activity.zero_()

    def forward(
        self,
        x: torch.Tensor,
        add_noise: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Forward pass - outputs action chunk.

        Args:
            x: Input from PSA [input_dim]
            add_noise: Whether to add exploration noise

        Returns:
            first_action: First action in chunk [action_dim] (for execution)
            action_chunk: Full chunk [chunk_size, action_dim] (for planning)
            info: Dict with activities
        """
        if x.dim() == 2:
            x = x.squeeze(0)

        # Store pre-synaptic activity
        self.pre_activity = x.detach()

        # Hidden layer
        hidden = F.relu(F.linear(x, self.w1, self.b1))
        self.hidden_activity = hidden.detach()

        # Output layer (tanh for bounded actions)
        output = torch.tanh(F.linear(hidden, self.w2, self.b2))
        self.output_activity = output.detach()

        # Reshape to [chunk_size, action_dim]
        action_chunk = output.view(self.chunk_size, self.action_dim)

        # Add noise during training
        if add_noise and self.training:
            noise = torch.randn_like(action_chunk) * self.noise_std
            action_chunk = torch.clamp(action_chunk + noise, -1, 1)

        # First action for receding horizon execution
        first_action = action_chunk[0]

        info = {
            'pre_activity': self.pre_activity,
            'hidden_activity': self.hidden_activity,
            'full_chunk': action_chunk,
        }

        return first_action, action_chunk, info

    def local_update(self, target_chunk: torch.Tensor):
        """
        Apply 3-factor learning rule to action chunk.

        The error is computed for the full chunk but weighted toward
        earlier steps (which have more immediate effect).

        Args:
            target_chunk: Demo actions [chunk_size, action_dim]
        """
        if target_chunk.dim() == 1:
            # Single action provided, replicate for chunk
            target_chunk = target_chunk.unsqueeze(0).expand(self.chunk_size, -1)

        # Ensure correct shape
        if target_chunk.shape[0] < self.chunk_size:
            # Pad with last action if not enough steps
            pad_size = self.chunk_size - target_chunk.shape[0]
            target_chunk = torch.cat([
                target_chunk,
                target_chunk[-1:].expand(pad_size, -1)
            ], dim=0)

        # Flatten target to match output
        target_flat = target_chunk.view(-1)

        # Compute per-step errors
        output_chunk = self.output_activity.view(self.chunk_size, self.action_dim)
        step_errors = target_chunk - output_chunk  # [chunk_size, action_dim]

        # Weight errors by temporal importance
        weighted_errors = step_errors * self.temporal_weights.unsqueeze(1)

        # Flatten weighted errors
        error_flat = weighted_errors.view(-1)
        error_flat = torch.clamp(error_flat, -1, 1)

        with torch.no_grad():
            # Layer 2 update: Δw2 = lr * error ⊗ hidden
            delta2 = self.lr * torch.outer(error_flat, self.hidden_activity)
            delta2 = torch.clamp(delta2, -0.1, 0.1)
            self.w2.data += delta2
            self.b2.data += self.lr * error_flat * 0.1

            # Layer 1 update: backprop error through layer 2
            error_hidden = F.linear(error_flat, self.w2.t())
            error_hidden = error_hidden * (self.hidden_activity > 0).float()
            error_hidden = torch.clamp(error_hidden, -1, 1)

            delta1 = self.lr * torch.outer(error_hidden, self.pre_activity)
            delta1 = torch.clamp(delta1, -0.1, 0.1)
            self.w1.data += delta1
            self.b1.data += self.lr * error_hidden * 0.1

            # Weight decay
            self.w1.data *= 0.9999
            self.w2.data *= 0.9999

            # Clamp weights
            self.w1.data.clamp_(-3, 3)
            self.w2.data.clamp_(-3, 3)


class PSAWithChunkedReadout(nn.Module):
    """
    PSA system with chunked action readout.

    Uses receding horizon: outputs chunk, executes first action, replans.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        chunk_size: int = 5,
        psa_neurons: int = 64,
        readout_hidden: int = 64,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

        from psa.predictive_psa import PredictivePSALayer

        self.psa = PredictivePSALayer(
            obs_dim=obs_dim,
            action_dim=action_dim,
            num_neurons=psa_neurons,
        )

        self.readout = ChunkedActionReadout(
            input_dim=psa_neurons,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dim=readout_hidden,
        )

        self.register_buffer('last_action', torch.zeros(action_dim))

        # Store current chunk for open-loop execution option
        self.current_chunk = None
        self.chunk_idx = 0

    def reset(self):
        """Reset all state."""
        self.psa.reset()
        self.readout.reset()
        self.last_action.zero_()
        self.current_chunk = None
        self.chunk_idx = 0

    def forward(
        self,
        obs: torch.Tensor,
        receding_horizon: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Forward pass.

        Args:
            obs: Current observation [obs_dim]
            receding_horizon: If True, replan every step. If False, execute chunk.

        Returns:
            action: Action to execute [action_dim]
            prediction: Predicted next observation [obs_dim]
            info: Debug info including full chunk
        """
        if receding_horizon or self.current_chunk is None or self.chunk_idx >= self.chunk_size:
            # Replan: get new chunk
            spikes, prediction, psa_info = self.psa(obs, self.last_action)
            action, chunk, readout_info = self.readout(spikes, add_noise=self.training)

            self.current_chunk = chunk
            self.chunk_idx = 0

            info = {
                'psa_info': psa_info,
                'readout_info': readout_info,
                'spikes': spikes,
                'chunk': chunk,
                'replanned': True,
            }
        else:
            # Open-loop: execute next action in chunk
            action = self.current_chunk[self.chunk_idx]
            self.chunk_idx += 1

            # Still need PSA forward for world model
            spikes, prediction, psa_info = self.psa(obs, self.last_action)

            info = {
                'psa_info': psa_info,
                'spikes': spikes,
                'chunk': self.current_chunk,
                'replanned': False,
            }

        self.last_action = action.detach().clone()
        return action, prediction, info

    def update(
        self,
        next_obs: torch.Tensor,
        demo_chunk: Optional[torch.Tensor] = None,
    ):
        """
        Local learning update.

        Args:
            next_obs: Actual next observation
            demo_chunk: Demo actions for chunk [chunk_size, action_dim] or single [action_dim]
        """
        self.psa.local_update(next_obs)

        if demo_chunk is not None:
            self.readout.local_update(demo_chunk)

    def consolidate(self):
        """Consolidate learned patterns."""
        self.psa.consolidate()
