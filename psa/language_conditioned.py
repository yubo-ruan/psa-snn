"""
Language-Conditioned PSA.

Key insight: Language should gate which assemblies are active, not just
concatenate to input. This allows the same observation to produce different
actions based on instruction.

Three mechanisms for language conditioning (all local, no backprop):
1. WTA bias: Language biases which assemblies win competition
2. Plasticity gain: Language modulates learning rate per module
3. Context spikes: Language adds context to observation encoding

Success criterion: Same scene + different instruction → different action.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, List
import math


class LanguageEncoder(nn.Module):
    """
    Simple language encoder using random projection.

    For LIBERO, we have ~100 task instructions. We can either:
    1. Random projection (no learning needed)
    2. Frozen pretrained embeddings (CLIP, etc.)
    3. Learned lookup (if we allow some gradient descent)

    This implementation uses random projection for bio-plausibility.
    """

    def __init__(
        self,
        vocab_size: int = 1000,
        embed_dim: int = 64,
        output_dim: int = 32,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.output_dim = output_dim

        # Random but fixed embeddings (no learning)
        self.register_buffer(
            'embeddings',
            torch.randn(vocab_size, embed_dim) / math.sqrt(embed_dim)
        )

        # Random projection to output dim
        self.register_buffer(
            'projection',
            torch.randn(embed_dim, output_dim) / math.sqrt(embed_dim)
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Encode language instruction.

        Args:
            token_ids: Token indices [seq_len] or [batch, seq_len]

        Returns:
            encoding: Language vector [output_dim] or [batch, output_dim]
        """
        squeeze = token_ids.dim() == 1
        if squeeze:
            token_ids = token_ids.unsqueeze(0)

        # Lookup embeddings
        embeds = self.embeddings[token_ids]  # [batch, seq_len, embed_dim]

        # Mean pooling over sequence
        pooled = embeds.mean(dim=1)  # [batch, embed_dim]

        # Project to output dim
        output = pooled @ self.projection  # [batch, output_dim]

        if squeeze:
            output = output.squeeze(0)

        return output

    def encode_text(self, text: str) -> torch.Tensor:
        """
        Simple text encoding using hash-based tokenization.

        For real use, replace with proper tokenizer.
        """
        # Simple hash-based "tokenization"
        words = text.lower().split()
        token_ids = torch.tensor([hash(w) % self.vocab_size for w in words])
        return self.forward(token_ids)


class LanguageGatedPSALayer(nn.Module):
    """
    PSA layer with language-based gating.

    Language conditions the layer through:
    1. Excitability bias: Changes threshold per neuron based on language
    2. Plasticity gain: Modulates learning rate based on language
    3. Context input: Language adds to input currents
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        language_dim: int = 32,
        num_neurons: int = 128,
        beta: float = 0.9,
        threshold: float = 1.0,
        fast_lr: float = 0.02,
        sparsity_target: float = 0.05,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.language_dim = language_dim
        self.num_neurons = num_neurons
        self.beta = beta
        self.fast_lr = fast_lr
        self.sparsity_target = sparsity_target

        input_dim = obs_dim + action_dim

        # Base threshold (learnable for homeostasis)
        self.threshold = nn.Parameter(torch.ones(num_neurons) * threshold)

        # Encoder: (obs, action) → currents
        self.encoder = nn.Parameter(torch.randn(num_neurons, input_dim) * 0.1)

        # Language → threshold bias (gating mechanism)
        # This is the key: language changes WHICH neurons fire
        self.lang_to_threshold = nn.Parameter(
            torch.randn(num_neurons, language_dim) * 0.1
        )

        # Language → plasticity gain
        self.lang_to_plasticity = nn.Parameter(
            torch.randn(num_neurons, language_dim) * 0.1
        )

        # Language → context current (additive)
        self.lang_to_context = nn.Parameter(
            torch.randn(num_neurons, language_dim) * 0.1
        )

        # Predictor: spikes → predicted next obs
        self.fast_predictor = nn.Parameter(torch.randn(obs_dim, num_neurons) * 0.1)
        self.register_buffer('slow_predictor', torch.zeros(obs_dim, num_neurons))
        self.predictor_bias = nn.Parameter(torch.zeros(obs_dim))

        # State buffers
        self.register_buffer('membrane', torch.zeros(num_neurons))
        self.register_buffer('spike_history', torch.zeros(num_neurons, 10))
        self.register_buffer('last_prediction', torch.zeros(obs_dim))
        self.register_buffer('current_language', torch.zeros(language_dim))
        self.has_prediction = False

        # Error tracking
        self.register_buffer('error_history', torch.zeros(100))
        self.error_idx = 0

    def reset(self):
        """Reset state for new episode."""
        self.membrane.zero_()
        self.spike_history.zero_()
        self.last_prediction.zero_()
        self.has_prediction = False
        self.error_history.zero_()
        self.error_idx = 0

    def set_language(self, language: torch.Tensor):
        """Set language context for this episode/segment."""
        self.current_language = language.detach()

    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        language: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Forward pass with language conditioning.
        """
        if obs.dim() == 2:
            obs = obs.squeeze(0)
        if action.dim() == 2:
            action = action.squeeze(0)

        # Use provided language or stored context
        if language is not None:
            self.current_language = language.detach()
        lang = self.current_language

        # Compute language-dependent threshold bias
        # Positive bias = harder to fire, negative = easier
        # STRONGER gating - language significantly changes which neurons fire
        threshold_bias = F.linear(lang, self.lang_to_threshold)
        effective_threshold = self.threshold + 1.5 * torch.tanh(threshold_bias)
        effective_threshold = effective_threshold.clamp(0.1, 5.0)

        # Language context current (STRONGER effect)
        context_current = F.linear(lang, self.lang_to_context)

        # Encode observation + action
        x = torch.cat([obs, action])
        input_current = F.linear(x, self.encoder)

        # Total current = input + language context (language has significant effect)
        total_current = input_current + 0.8 * context_current

        # LIF dynamics
        self.membrane = self.beta * self.membrane + total_current

        # Spike generation with language-modulated threshold
        spikes = (self.membrane >= effective_threshold).float()

        # Soft reset
        self.membrane = self.membrane - spikes * effective_threshold

        # Update spike history
        self.spike_history = torch.roll(self.spike_history, -1, dims=1)
        self.spike_history[:, -1] = spikes

        # Predict next observation
        effective_pred = self.fast_predictor + self.slow_predictor
        prediction = F.linear(spikes, effective_pred) + self.predictor_bias

        # Compute prediction error if we have previous prediction
        error = torch.zeros(self.obs_dim, device=obs.device)
        if self.has_prediction:
            error = obs - self.last_prediction
            self._update_error_history(error)

        self.last_prediction = prediction.detach().clone()
        self.has_prediction = True

        info = {
            'error': error,
            'firing_rate': spikes.mean().item(),
            'effective_threshold': effective_threshold.mean().item(),
        }

        return spikes, prediction, info

    def _update_error_history(self, error: torch.Tensor):
        error_mag = error.abs().mean().item()
        self.error_history[self.error_idx % 100] = error_mag
        self.error_idx += 1

    def _compute_surprise(self) -> float:
        if self.error_idx < 5:
            return 0.5
        history = self.error_history[:min(self.error_idx, 100)]
        mean_error = history.mean().item()
        std_error = history.std().item() + 1e-6
        current_error = self.error_history[(self.error_idx - 1) % 100].item()
        z_score = (current_error - mean_error) / std_error
        return float(torch.sigmoid(torch.tensor(z_score)).item())

    def local_update(self, actual_next_obs: Optional[torch.Tensor] = None):
        """
        Local plasticity with language-modulated learning rate.
        """
        if not self.has_prediction:
            return

        if actual_next_obs is not None:
            error = actual_next_obs.detach().float() - self.last_prediction.detach()
        else:
            if self.error_idx < 1:
                return
            error_mag = self.error_history[(self.error_idx - 1) % 100].item()
            error = torch.ones(self.obs_dim, device=self.membrane.device) * error_mag * 0.1

        error = torch.clamp(error, -2.0, 2.0)

        # Language-modulated plasticity gain
        plasticity_gain = F.linear(self.current_language, self.lang_to_plasticity)
        plasticity_gain = 1.0 + 0.5 * torch.tanh(plasticity_gain)  # [0.5, 1.5]

        surprise = self._compute_surprise()
        base_lr = self.fast_lr * max(surprise, 0.2)

        spike_rate = self.spike_history.mean(dim=1)

        with torch.no_grad():
            # Modulated learning rate per synapse
            # delta[i,j] = base_lr * plasticity_gain[j] * error[i] * spike_rate[j]
            modulated_rate = spike_rate * plasticity_gain
            delta = base_lr * torch.outer(error, modulated_rate)
            delta = torch.clamp(delta, -0.05, 0.05)

            self.fast_predictor.data += delta
            self.fast_predictor.data *= 0.999
            self.fast_predictor.data.clamp_(-5.0, 5.0)

            # Homeostatic threshold adjustment
            current_rate = spike_rate.mean()
            threshold_delta = 0.001 * (current_rate - self.sparsity_target)
            self.threshold.data += threshold_delta
            self.threshold.data.clamp_(0.1, 5.0)

    def consolidate(self):
        """Move stable patterns to slow weights."""
        with torch.no_grad():
            self.slow_predictor += 0.005 * (self.fast_predictor - self.slow_predictor)


class LanguageConditionedPSA(nn.Module):
    """
    Complete language-conditioned PSA system.

    Architecture:
    - Language encoder (frozen random projection)
    - Language-gated PSA layer
    - 3-factor action readout

    The key innovation: same observation + different language → different behavior
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        num_neurons: int = 128,
        language_dim: int = 32,
        readout_hidden: int = 64,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.language_dim = language_dim

        # Language encoder
        self.language_encoder = LanguageEncoder(
            vocab_size=1000,
            embed_dim=64,
            output_dim=language_dim,
        )

        # Language-gated PSA
        self.psa = LanguageGatedPSALayer(
            obs_dim=obs_dim,
            action_dim=action_dim,
            language_dim=language_dim,
            num_neurons=num_neurons,
        )

        # Action readout (from psa.action_readout)
        from psa.action_readout import ThreeFactorReadout
        self.readout = ThreeFactorReadout(
            input_dim=num_neurons,
            action_dim=action_dim,
            hidden_dim=readout_hidden,
        )

        # Store last action
        self.register_buffer('last_action', torch.zeros(action_dim))
        self.register_buffer('current_language', torch.zeros(language_dim))

    def reset(self):
        """Reset for new episode."""
        self.psa.reset()
        self.readout.reset()
        self.last_action.zero_()

    def set_task(self, instruction: str):
        """Set task instruction for the episode."""
        self.current_language = self.language_encoder.encode_text(instruction)
        self.psa.set_language(self.current_language)

    def set_task_embedding(self, language: torch.Tensor):
        """Set task via pre-computed embedding."""
        self.current_language = language.detach()
        self.psa.set_language(self.current_language)

    def forward(
        self,
        obs: torch.Tensor,
        language: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Forward pass.

        Args:
            obs: Current observation [obs_dim]
            language: Optional language embedding (uses stored if None)

        Returns:
            action: Predicted action [action_dim]
            prediction: Predicted next observation [obs_dim]
            info: Debug info
        """
        if language is not None:
            self.current_language = language.detach()

        # PSA forward
        spikes, prediction, psa_info = self.psa(
            obs, self.last_action, self.current_language
        )

        # Action readout
        action, readout_info = self.readout(spikes, add_noise=self.training)

        self.last_action = action.detach().clone()

        info = {
            'psa_info': psa_info,
            'readout_info': readout_info,
            'spikes': spikes,
            'language': self.current_language,
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
