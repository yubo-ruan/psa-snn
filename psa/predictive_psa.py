"""
Predictive PSA - Neurons predict next sensory input, not neighbor spikes.

Key insight: The original PSA predicted neighbor firing rates, but this doesn't
force neurons to encode task-relevant information. Instead, we should predict
the next observation (or its encoding), conditioned on action.

Architecture:
    Layer L receives: current_obs_encoding + action
    Layer L predicts: next_obs_encoding (at layer L)
    Error: predicted_next - actual_next
    Update: local Hebbian + error modulation

This aligns the representation with dynamics learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, List
import math


class PredictivePSALayer(nn.Module):
    """
    PSA layer that predicts next observation encoding.

    Each neuron:
    1. Receives (obs_encoding, action) as input
    2. Fires based on LIF dynamics
    3. Collectively predicts next obs_encoding
    4. Updates weights based on prediction error

    The key difference from original PSA:
    - OLD: predict neighbor spike rates (self-referential)
    - NEW: predict next sensory input (externally grounded)
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        num_neurons: int = 128,
        beta: float = 0.9,
        threshold: float = 1.0,
        fast_lr: float = 0.02,  # Moderate learning rate
        slow_lr: float = 0.005,
        consolidation_rate: float = 0.01,  # Faster consolidation for stability
        sparsity_target: float = 0.05,  # Sparser representation
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_neurons = num_neurons
        self.beta = beta
        self.fast_lr = fast_lr
        self.slow_lr = slow_lr
        self.consolidation_rate = consolidation_rate
        self.sparsity_target = sparsity_target

        input_dim = obs_dim + action_dim

        # Learnable threshold per neuron (for homeostasis)
        self.threshold = nn.Parameter(torch.ones(num_neurons) * threshold)

        # Encoder: (obs, action) → currents
        self.encoder = nn.Parameter(torch.randn(num_neurons, input_dim) * 0.1)

        # Predictor: spikes → predicted next obs
        # This is the key: we predict external observation, not internal spikes
        self.fast_predictor = nn.Parameter(torch.randn(obs_dim, num_neurons) * 0.1)
        self.register_buffer('slow_predictor', torch.zeros(obs_dim, num_neurons))
        self.predictor_bias = nn.Parameter(torch.zeros(obs_dim))

        # Membrane potential state
        self.register_buffer('membrane', torch.zeros(num_neurons))

        # Recent spike history (for rate estimation)
        self.register_buffer('spike_history', torch.zeros(num_neurons, 10))

        # Error tracking for surprise computation
        self.register_buffer('error_history', torch.zeros(100))
        self.error_idx = 0

        # Last prediction (for computing error on next step)
        self.register_buffer('last_prediction', torch.zeros(obs_dim))
        self.has_prediction = False

    def reset(self):
        """Reset state for new episode."""
        self.membrane.zero_()
        self.spike_history.zero_()
        self.last_prediction.zero_()
        self.has_prediction = False
        self.error_history.zero_()
        self.error_idx = 0

    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Forward pass.

        Args:
            obs: Current observation encoding [obs_dim]
            action: Action [action_dim]

        Returns:
            spikes: Neuron spikes [num_neurons]
            prediction: Predicted next obs [obs_dim]
            info: Dict with error, surprise, etc.
        """
        # Ensure proper dimensions
        if obs.dim() == 2:
            obs = obs.squeeze(0)
        if action.dim() == 2:
            action = action.squeeze(0)

        # Compute prediction error from last step
        error = torch.zeros(self.obs_dim, device=obs.device)
        if self.has_prediction:
            error = obs - self.last_prediction
            self._update_error_history(error)

        # Encode input
        x = torch.cat([obs, action])
        currents = F.linear(x, self.encoder)

        # LIF dynamics
        self.membrane = self.beta * self.membrane + currents

        # Spike generation with soft threshold
        spikes = (self.membrane >= self.threshold).float()

        # Soft reset
        self.membrane = self.membrane - spikes * self.threshold

        # Update spike history
        self.spike_history = torch.roll(self.spike_history, -1, dims=1)
        self.spike_history[:, -1] = spikes

        # Predict next observation
        effective_pred = self.fast_predictor + self.slow_predictor
        prediction = F.linear(spikes, effective_pred) + self.predictor_bias

        # Store prediction for next step's error
        self.last_prediction = prediction.detach().clone()
        self.has_prediction = True

        # Compute surprise
        surprise = self._compute_surprise()

        info = {
            'error': error,
            'surprise': surprise,
            'firing_rate': spikes.mean().item(),
            'membrane_mean': self.membrane.mean().item(),
        }

        return spikes, prediction, info

    def _update_error_history(self, error: torch.Tensor):
        """Track prediction errors for surprise computation."""
        error_mag = error.abs().mean().item()
        self.error_history[self.error_idx % 100] = error_mag
        self.error_idx += 1

    def _compute_surprise(self) -> float:
        """Compute surprise as normalized prediction error."""
        if self.error_idx < 5:
            return 0.5  # Not enough history

        history = self.error_history[:min(self.error_idx, 100)]
        mean_error = history.mean().item()
        std_error = history.std().item() + 1e-6

        current_error = self.error_history[(self.error_idx - 1) % 100].item()
        z_score = (current_error - mean_error) / std_error

        return float(torch.sigmoid(torch.tensor(z_score)).item())

    def local_update(self, actual_next_obs: Optional[torch.Tensor] = None):
        """
        Local plasticity update based on prediction error.

        The update rule:
        Δw_predictor = lr * error ⊗ spike_rate * surprise

        This is Hebbian (spikes) + error-modulated + surprise-gated.
        """
        if not self.has_prediction:
            return

        # Get error from stored prediction vs current obs
        if actual_next_obs is not None:
            error = actual_next_obs.detach().float() - self.last_prediction.detach()
        else:
            # Use stored error magnitude (less accurate but works)
            if self.error_idx < 1:
                return
            error_mag = self.error_history[(self.error_idx - 1) % 100].item()
            error = torch.ones(self.obs_dim, device=self.membrane.device) * error_mag * 0.1

        # Clamp error to prevent explosion (but allow reasonable range)
        error = torch.clamp(error, -2.0, 2.0)

        surprise = self._compute_surprise()
        effective_lr = self.fast_lr * max(surprise, 0.2)  # Stronger minimum learning

        # Use average spike rate over window (more stable than single timestep)
        spike_rate = self.spike_history.mean(dim=1)  # [num_neurons]

        with torch.no_grad():
            # Update predictor weights
            # delta[i,j] = lr * error[i] * spike_rate[j]
            delta = effective_lr * torch.outer(error, spike_rate)

            # Clamp delta
            delta = torch.clamp(delta, -0.05, 0.05)

            self.fast_predictor.data += delta

            # Stronger weight decay for generalization
            self.fast_predictor.data *= 0.999

            # Anti-Hebbian drift: decay toward slow weights
            self.fast_predictor.data = (
                (1 - self.consolidation_rate) * self.fast_predictor.data +
                self.consolidation_rate * self.slow_predictor
            )

            # Clamp weights to prevent explosion
            self.fast_predictor.data.clamp_(-5.0, 5.0)

            # Homeostatic threshold adjustment
            current_rate = spike_rate.mean()
            threshold_delta = 0.001 * (current_rate - self.sparsity_target)
            self.threshold.data += threshold_delta
            self.threshold.data.clamp_(0.1, 5.0)

    def consolidate(self):
        """Move stable patterns to slow weights."""
        with torch.no_grad():
            self.slow_predictor += self.slow_lr * (self.fast_predictor - self.slow_predictor)


class PredictivePSANetwork(nn.Module):
    """
    Multi-layer Predictive PSA Network.

    Architecture:
    - Layer 0: (raw_obs, action) → spikes_0 → pred_obs_0
    - Layer 1: (spikes_0, action) → spikes_1 → pred_obs_1 (slower dynamics)
    - ...

    Each layer predicts the next observation at its abstraction level.
    Higher layers capture slower dynamics (larger receptive fields in time).
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        layer_sizes: List[int] = [128, 64],
        beta: float = 0.9,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # Build layers
        self.layers = nn.ModuleList()

        # First layer takes raw obs
        self.layers.append(PredictivePSALayer(
            obs_dim=obs_dim,
            action_dim=action_dim,
            num_neurons=layer_sizes[0],
            beta=beta,
        ))

        # Subsequent layers take previous layer spikes as "obs"
        for i in range(1, len(layer_sizes)):
            self.layers.append(PredictivePSALayer(
                obs_dim=layer_sizes[i-1],  # Previous layer spikes as input
                action_dim=action_dim,
                num_neurons=layer_sizes[i],
                beta=beta,
            ))

        # Store last action for higher layer prediction
        self.register_buffer('last_action', torch.zeros(action_dim))

    def reset(self):
        """Reset all layers."""
        for layer in self.layers:
            layer.reset()

    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], Dict]:
        """
        Forward pass through all layers.

        Args:
            obs: Current observation [obs_dim]
            action: Current action [action_dim]

        Returns:
            final_spikes: Top layer spikes
            all_predictions: List of predictions at each layer
            info: Aggregated info
        """
        all_spikes = []
        all_predictions = []
        all_errors = []
        all_surprises = []

        current_input = obs
        for i, layer in enumerate(self.layers):
            spikes, prediction, layer_info = layer(current_input, action)
            all_spikes.append(spikes)
            all_predictions.append(prediction)
            all_errors.append(layer_info['error'])
            all_surprises.append(layer_info['surprise'])

            # Next layer receives these spikes as input
            current_input = spikes

        info = {
            'all_spikes': all_spikes,
            'errors': all_errors,
            'surprises': all_surprises,
            'mean_surprise': sum(all_surprises) / len(all_surprises),
            'mean_error': sum(e.abs().mean().item() for e in all_errors) / len(all_errors),
        }

        return all_spikes[-1], all_predictions, info

    def local_update(self, actual_next_obs: Optional[torch.Tensor] = None):
        """Update all layers with actual next observation."""
        # First layer predicts next raw observation
        # Higher layers predict next spike patterns (their "observation")
        for i, layer in enumerate(self.layers):
            if i == 0 and actual_next_obs is not None:
                layer.local_update(actual_next_obs)
            else:
                layer.local_update()

    def consolidate(self):
        """Consolidate all layers."""
        for layer in self.layers:
            layer.consolidate()

    def predict_next_obs(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, float]:
        """
        Predict next observation given current obs and action.

        Args:
            obs: Current observation
            action: Action to take

        Returns:
            predicted_next_obs: Predicted next observation
            uncertainty: Prediction uncertainty
        """
        _, predictions, info = self.forward(obs, action)

        # First layer predicts next raw observation
        predicted_next = predictions[0]
        uncertainty = info['mean_surprise']

        return predicted_next, uncertainty


class PredictivePSAWorldModel(nn.Module):
    """
    Complete world model using Predictive PSA.

    Combines:
    1. Frozen encoder for observations
    2. Predictive PSA for dynamics learning
    3. Optional language conditioning
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        latent_dim: int = 64,
        psa_layers: List[int] = [128, 64],
        language_dim: Optional[int] = None,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim

        # Observation encoder (could be frozen pretrained)
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
        )

        # Language conditioning (optional)
        self.use_language = language_dim is not None
        if self.use_language:
            self.language_proj = nn.Linear(language_dim, latent_dim)

        # Effective action dim (might include language)
        effective_action_dim = action_dim
        if self.use_language:
            effective_action_dim += latent_dim

        # Predictive PSA
        self.psa = PredictivePSANetwork(
            obs_dim=latent_dim,
            action_dim=effective_action_dim,
            layer_sizes=psa_layers,
        )

        # Decoder: latent → obs (for evaluation)
        self.obs_decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, obs_dim),
        )

    def reset(self):
        """Reset PSA state."""
        self.psa.reset()

    def encode_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode observation to latent space."""
        return self.obs_encoder(obs)

    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        language: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Forward pass.

        Args:
            obs: Raw observation [obs_dim]
            action: Action [action_dim]
            language: Optional language embedding [language_dim]

        Returns:
            predicted_next_obs: Predicted next observation [obs_dim]
            latent_prediction: Latent space prediction [latent_dim]
            info: Debug info
        """
        # Encode observation
        latent = self.encode_obs(obs.float())

        # Prepare action (possibly with language)
        if self.use_language and language is not None:
            lang_latent = self.language_proj(language.float())
            action_combined = torch.cat([action.float(), lang_latent])
        else:
            action_combined = action.float()

        # PSA forward
        spikes, predictions, info = self.psa(latent, action_combined)

        # Decode prediction to observation space
        latent_prediction = predictions[0]  # First layer predicts latent
        predicted_next_obs = self.obs_decoder(latent_prediction)

        return predicted_next_obs, latent_prediction, info

    def update(self, next_obs: Optional[torch.Tensor] = None):
        """
        Update PSA with local learning.

        Args:
            next_obs: Actual next observation (for computing prediction error)
        """
        # Encode next observation for the PSA layer
        if next_obs is not None:
            next_latent = self.encode_obs(next_obs.float()).detach()
            self.psa.local_update(next_latent)
        else:
            self.psa.local_update()

    def consolidate(self):
        """Consolidate learned patterns."""
        self.psa.consolidate()

    def prediction_error(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        next_obs: torch.Tensor,
        language: Optional[torch.Tensor] = None,
    ) -> Tuple[float, float]:
        """
        Compute prediction error for evaluation.

        Returns:
            obs_mse: MSE in observation space
            latent_mse: MSE in latent space
        """
        with torch.no_grad():
            predicted_next, latent_pred, _ = self.forward(obs, action, language)

            obs_mse = F.mse_loss(predicted_next, next_obs.float()).item()

            actual_latent = self.encode_obs(next_obs.float())
            latent_mse = F.mse_loss(latent_pred, actual_latent).item()

        return obs_mse, latent_mse
