"""
LIBERO-specific PSA adaptations.

Key changes from vanilla PSA:
1. Frozen vision encoder (not raw pixels)
2. Frame differencing for event-like input
3. Language-conditioned gating
4. 2 timescales (50ms fast, 250ms slow)
5. Modular columns with instruction routing
6. Homeostatic burn-in for calibration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional
import numpy as np

from .network import PSANetwork
from .neuron import PSALayer


class FrozenVisionEncoder(nn.Module):
    """
    Frozen visual frontend for PSA.

    Options:
    1. Pretrained ResNet/ViT features (no training)
    2. Frame differencing (DVS-like events from RGB)

    This gives PSA a stable "sensory nerve" - we're not training
    the vision part, just like LIBERO baselines use pretrained backbones.
    """

    def __init__(
        self,
        mode: str = "resnet18",  # "resnet18", "frame_diff", "both"
        output_dim: int = 256,
        frame_diff_threshold: float = 0.1,
    ):
        super().__init__()
        self.mode = mode
        self.output_dim = output_dim
        self.frame_diff_threshold = frame_diff_threshold

        if mode in ["resnet18", "both"]:
            # Frozen ResNet18 backbone
            import torchvision.models as models
            resnet = models.resnet18(pretrained=True)
            # Remove final FC layer
            self.cnn = nn.Sequential(*list(resnet.children())[:-1])
            # Freeze
            for param in self.cnn.parameters():
                param.requires_grad = False
            self.cnn_proj = nn.Linear(512, output_dim // 2 if mode == "both" else output_dim)

        if mode in ["frame_diff", "both"]:
            # Frame differencing produces sparse "events"
            self.prev_frame = None
            self.diff_proj = nn.Linear(128 * 128 * 3, output_dim // 2 if mode == "both" else output_dim)

    def compute_frame_diff(self, frame: torch.Tensor) -> torch.Tensor:
        """
        DVS-like frame differencing.

        Returns sparse "events" where intensity changed significantly.
        """
        if self.prev_frame is None:
            self.prev_frame = frame
            return torch.zeros_like(frame)

        # Compute difference
        diff = frame - self.prev_frame

        # Threshold to get sparse events
        events = torch.where(
            diff.abs() > self.frame_diff_threshold,
            torch.sign(diff),
            torch.zeros_like(diff)
        )

        self.prev_frame = frame
        return events

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Encode image to low-dim tokens.

        Args:
            image: RGB image [B, 3, H, W] or [3, H, W]

        Returns:
            tokens: Visual tokens [B, output_dim] or [output_dim]
        """
        squeeze = False
        if image.dim() == 3:
            image = image.unsqueeze(0)
            squeeze = True

        features = []

        if self.mode in ["resnet18", "both"]:
            with torch.no_grad():
                cnn_feat = self.cnn(image).flatten(1)
            cnn_feat = self.cnn_proj(cnn_feat)
            features.append(cnn_feat)

        if self.mode in ["frame_diff", "both"]:
            # Compute frame diff for each batch item
            diff_feats = []
            for i in range(image.shape[0]):
                diff = self.compute_frame_diff(image[i])
                diff_feats.append(diff.flatten())
            diff_feat = torch.stack(diff_feats)
            diff_feat = self.diff_proj(diff_feat)
            features.append(diff_feat)

        output = torch.cat(features, dim=-1) if len(features) > 1 else features[0]

        if squeeze:
            output = output.squeeze(0)

        return output

    def reset(self):
        """Reset frame differencing state."""
        self.prev_frame = None


class LanguageGating(nn.Module):
    """
    Language-conditioned gating for PSA.

    The instruction gates:
    1. Plasticity (learning rate modulation)
    2. Competition (WTA priors / bias)
    3. Module routing (which PSA modules are active)

    Without this, assemblies mix tasks that share visuals but differ in goals.
    """

    def __init__(
        self,
        language_dim: int = 384,  # CLIP/sentence-transformer dim
        hidden_dim: int = 128,
        num_modules: int = 8,
    ):
        super().__init__()
        self.num_modules = num_modules

        # Language encoder (frozen CLIP or sentence-transformer)
        self.language_proj = nn.Sequential(
            nn.Linear(language_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Plasticity gate: language → learning rate multiplier per module
        self.plasticity_gate = nn.Sequential(
            nn.Linear(hidden_dim, num_modules),
            nn.Sigmoid(),  # 0-1 multiplier
        )

        # Competition bias: language → WTA bias per module
        self.competition_bias = nn.Linear(hidden_dim, num_modules)

        # Module routing: language → soft routing weights
        self.routing = nn.Sequential(
            nn.Linear(hidden_dim, num_modules),
            nn.Softmax(dim=-1),
        )

        # Cache for current instruction
        self.cached_instruction = None
        self.cached_gates = None

    def encode_instruction(self, instruction_embedding: torch.Tensor):
        """
        Encode instruction and cache gates.

        Args:
            instruction_embedding: Pre-computed language embedding [language_dim]
        """
        if instruction_embedding.dim() == 1:
            instruction_embedding = instruction_embedding.unsqueeze(0)

        hidden = self.language_proj(instruction_embedding)

        self.cached_gates = {
            'plasticity': self.plasticity_gate(hidden).squeeze(0),
            'bias': self.competition_bias(hidden).squeeze(0),
            'routing': self.routing(hidden).squeeze(0),
        }
        self.cached_instruction = instruction_embedding

    def get_plasticity_multiplier(self, module_idx: int) -> float:
        """Get learning rate multiplier for a module."""
        if self.cached_gates is None:
            return 1.0
        return self.cached_gates['plasticity'][module_idx].item()

    def get_competition_bias(self, module_idx: int) -> float:
        """Get WTA bias for a module."""
        if self.cached_gates is None:
            return 0.0
        return self.cached_gates['bias'][module_idx].item()

    def get_routing_weight(self, module_idx: int) -> float:
        """Get routing weight for a module."""
        if self.cached_gates is None:
            return 1.0 / self.num_modules
        return self.cached_gates['routing'][module_idx].item()


class HomeostaticCalibration:
    """
    Homeostatic burn-in for PSA.

    First few thousand steps: tune thresholds and inhibitory gains
    to hit target sparsity BEFORE enabling STDP.

    Without this, random connections latch onto junk correlations.
    """

    def __init__(
        self,
        target_sparsity: float = 0.05,  # 5% neurons active
        calibration_steps: int = 2000,
        threshold_lr: float = 0.01,
        inhibition_lr: float = 0.01,
    ):
        self.target_sparsity = target_sparsity
        self.calibration_steps = calibration_steps
        self.threshold_lr = threshold_lr
        self.inhibition_lr = inhibition_lr

        self.current_step = 0
        self.is_calibrating = True

        # Track firing rates
        self.firing_rates = {}

    def update(
        self,
        layer_name: str,
        spikes: torch.Tensor,
        layer: PSALayer,
    ):
        """
        Update thresholds to achieve target sparsity.

        Args:
            layer_name: Name of the layer
            spikes: Current spike pattern
            layer: PSA layer to adjust
        """
        self.current_step += 1

        if self.current_step > self.calibration_steps:
            self.is_calibrating = False
            return

        # Compute current sparsity
        current_sparsity = spikes.float().mean().item()

        # Track for monitoring
        if layer_name not in self.firing_rates:
            self.firing_rates[layer_name] = []
        self.firing_rates[layer_name].append(current_sparsity)

        # Adjust thresholds
        error = current_sparsity - self.target_sparsity

        for neuron in layer.neurons:
            # If firing too much, raise threshold
            # If firing too little, lower threshold
            neuron.threshold += self.threshold_lr * error
            neuron.threshold = max(0.1, min(5.0, neuron.threshold))  # Clamp

    def should_enable_plasticity(self) -> bool:
        """Check if calibration is complete."""
        return not self.is_calibrating

    def get_summary(self) -> str:
        """Get calibration summary."""
        lines = [f"Homeostatic Calibration: step {self.current_step}/{self.calibration_steps}"]
        for name, rates in self.firing_rates.items():
            if rates:
                recent = rates[-100:] if len(rates) > 100 else rates
                lines.append(f"  {name}: sparsity={np.mean(recent):.3f} (target={self.target_sparsity})")
        return "\n".join(lines)


class TwoTimescalePSA(nn.Module):
    """
    PSA with 2 timescales for LIBERO:

    Fast (L1): predicts next-step sensor tokens at 1× env step (50ms @ 20Hz)
    Slow (L2): predicts over action chunks (5 steps = 250ms)

    This matches what works for manipulation: chunking helps long-horizon control.
    """

    def __init__(
        self,
        input_dim: int,
        fast_neurons: int = 256,
        slow_neurons: int = 128,
        fast_window: int = 1,   # 50ms
        slow_window: int = 5,   # 250ms (action chunk)
        chunk_size: int = 5,
    ):
        super().__init__()
        self.chunk_size = chunk_size
        self.fast_window = fast_window
        self.slow_window = slow_window

        # Fast layer: step-by-step prediction
        self.fast_layer = PSALayer(
            num_neurons=fast_neurons,
            input_dim=input_dim,
            neighbor_radius=5,
            prediction_window=fast_window,
        )

        # Slow layer: chunk-level prediction
        self.slow_layer = PSALayer(
            num_neurons=slow_neurons,
            input_dim=fast_neurons,
            neighbor_radius=5,
            prediction_window=slow_window,
        )

        # Buffer for chunk accumulation
        self.register_buffer('chunk_buffer', torch.zeros(chunk_size, fast_neurons))
        self.chunk_idx = 0

        # Step counter for timescale separation
        self.step = 0

    def reset(self):
        """Reset both layers and buffers."""
        self.fast_layer.reset()
        self.slow_layer.reset()
        self.chunk_buffer.zero_()
        self.chunk_idx = 0
        self.step = 0

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Forward pass with 2 timescales.

        Fast layer runs every step.
        Slow layer runs every chunk_size steps.

        Returns:
            fast_spikes: Fast layer output
            slow_spikes: Slow layer output (or None if not chunk boundary)
            info: Prediction info
        """
        self.step += 1

        # Fast layer: always runs
        fast_spikes, fast_pred = self.fast_layer(x)

        # Accumulate into chunk buffer
        if fast_spikes.dim() == 1:
            self.chunk_buffer[self.chunk_idx] = fast_spikes
        else:
            self.chunk_buffer[self.chunk_idx] = fast_spikes[0]
        self.chunk_idx = (self.chunk_idx + 1) % self.chunk_size

        # Slow layer: runs at chunk boundaries
        slow_spikes = None
        slow_pred = None

        if self.step % self.chunk_size == 0:
            # Aggregate chunk (mean pooling)
            chunk_summary = self.chunk_buffer.mean(dim=0)
            slow_spikes, slow_pred = self.slow_layer(chunk_summary)

        info = {
            'fast_pred': fast_pred,
            'slow_pred': slow_pred,
            'fast_surprise': self.fast_layer.compute_surprise(),
            'slow_surprise': self.slow_layer.compute_surprise() if slow_spikes is not None else 0,
        }

        return fast_spikes, slow_spikes, info

    def local_update(self, language_gate: Optional[LanguageGating] = None):
        """Update both layers with optional language gating."""
        # Fast layer always updates
        fast_multiplier = 1.0
        if language_gate is not None:
            fast_multiplier = language_gate.get_plasticity_multiplier(0)

        if fast_multiplier > 0.1:  # Only update if gate is open
            for neuron in self.fast_layer.neurons:
                neuron.local_update(surprise=fast_multiplier)

        # Slow layer updates at chunk boundaries
        if self.step % self.chunk_size == 0:
            slow_multiplier = 1.0
            if language_gate is not None:
                slow_multiplier = language_gate.get_plasticity_multiplier(1)

            if slow_multiplier > 0.1:
                for neuron in self.slow_layer.neurons:
                    neuron.local_update(surprise=slow_multiplier)


class ModularPSA(nn.Module):
    """
    Modular PSA with instruction routing for LIBERO.

    Many small recurrent modules (columns) with sparse inter-module links.
    Language instruction selects which modules compete to form assemblies.

    This prevents cross-task interference in lifelong learning.
    """

    def __init__(
        self,
        input_dim: int,
        num_modules: int = 8,
        neurons_per_module: int = 64,
        inter_module_sparsity: float = 0.1,
    ):
        super().__init__()
        self.num_modules = num_modules
        self.neurons_per_module = neurons_per_module

        # Create modules (columns)
        self.psa_modules = nn.ModuleList([
            TwoTimescalePSA(
                input_dim=input_dim,
                fast_neurons=neurons_per_module,
                slow_neurons=neurons_per_module // 2,
            )
            for _ in range(num_modules)
        ])

        # Sparse inter-module connections
        self.inter_module = nn.ModuleList([
            nn.Linear(neurons_per_module, neurons_per_module, bias=False)
            for _ in range(num_modules)
        ])

        # Initialize inter-module connections as sparse
        for linear in self.inter_module:
            mask = torch.rand_like(linear.weight) < inter_module_sparsity
            linear.weight.data *= mask.float()

        # Language gating
        self.language_gate = LanguageGating(num_modules=num_modules)

        # Homeostatic calibration
        self.calibration = HomeostaticCalibration()

    def reset(self):
        """Reset all modules."""
        for module in self.psa_modules:
            module.reset()

    def forward(
        self,
        x: torch.Tensor,
        instruction_embedding: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass with modular routing.

        Args:
            x: Input tokens
            instruction_embedding: Language instruction embedding

        Returns:
            output: Aggregated module outputs
            info: Module-wise info
        """
        # Update language gates if instruction provided
        if instruction_embedding is not None:
            self.language_gate.encode_instruction(instruction_embedding)

        # Run each module with routing weight
        module_outputs = []
        info = {'module_activities': [], 'routing_weights': []}

        for i, module in enumerate(self.psa_modules):
            # Get routing weight
            weight = self.language_gate.get_routing_weight(i)
            info['routing_weights'].append(weight)

            # Skip if weight is too low
            if weight < 0.05:
                module_outputs.append(torch.zeros(self.neurons_per_module, device=x.device))
                info['module_activities'].append(0.0)
                continue

            # Run module
            fast_spikes, slow_spikes, module_info = module(x)

            # Homeostatic calibration
            if self.calibration.is_calibrating:
                self.calibration.update(f"module_{i}_fast", fast_spikes, module.fast_layer)

            # Weight by routing
            weighted_output = fast_spikes * weight
            module_outputs.append(weighted_output)
            info['module_activities'].append(fast_spikes.float().mean().item())

        # Aggregate (sparse competition via language routing)
        output = torch.stack(module_outputs).sum(dim=0)

        return output, info

    def local_update(self):
        """Update all modules with language-gated plasticity."""
        if not self.calibration.should_enable_plasticity():
            return  # Still calibrating

        for i, module in enumerate(self.psa_modules):
            module.local_update(self.language_gate)


class LIBEROPSAAgent(nn.Module):
    """
    Complete PSA agent for LIBERO.

    Combines:
    1. Frozen vision encoder
    2. Language-conditioned gating
    3. 2-timescale modular PSA
    4. Homeostatic calibration
    5. Active inference action selection
    """

    def __init__(
        self,
        image_size: Tuple[int, int] = (128, 128),
        proprio_dim: int = 7,
        action_dim: int = 7,
        language_dim: int = 384,
        num_modules: int = 8,
        neurons_per_module: int = 64,
    ):
        super().__init__()
        self.action_dim = action_dim

        # Vision encoder (frozen)
        self.vision = FrozenVisionEncoder(mode="both", output_dim=256)

        # Input: vision + proprio
        input_dim = 256 + proprio_dim

        # Modular PSA
        self.psa = ModularPSA(
            input_dim=input_dim,
            num_modules=num_modules,
            neurons_per_module=neurons_per_module,
        )

        # Action decoder (simple for now)
        self.action_decoder = nn.Sequential(
            nn.Linear(neurons_per_module, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh(),
        )

        # Store language embedding
        self.current_instruction = None

    def reset(self):
        """Reset for new episode."""
        self.vision.reset()
        self.psa.reset()

    def set_instruction(self, instruction_embedding: torch.Tensor):
        """Set current task instruction."""
        self.current_instruction = instruction_embedding

    def forward(
        self,
        image: torch.Tensor,
        proprio: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass.

        Args:
            image: RGB image [3, H, W]
            proprio: Proprioceptive state [proprio_dim]

        Returns:
            action: Predicted action [action_dim]
            info: Debug info
        """
        # Encode vision
        vision_tokens = self.vision(image)

        # Combine with proprio
        if proprio.dim() == 1:
            proprio = proprio.unsqueeze(0)
        if vision_tokens.dim() == 1:
            vision_tokens = vision_tokens.unsqueeze(0)

        x = torch.cat([vision_tokens, proprio], dim=-1).squeeze(0)

        # PSA forward
        psa_output, info = self.psa(x, self.current_instruction)

        # Decode action
        action = self.action_decoder(psa_output)

        return action, info

    def update(self):
        """Local plasticity update."""
        self.psa.local_update()

    def get_summary(self) -> str:
        """Get agent summary."""
        lines = ["LIBERO PSA Agent Summary"]
        lines.append(self.psa.calibration.get_summary())
        lines.append(f"Modules: {self.psa.num_modules}")
        lines.append(f"Neurons/module: {self.psa.neurons_per_module}")
        return "\n".join(lines)
