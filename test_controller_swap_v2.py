"""
Controller Swap Test V2 - With actual prediction error measurement.

Key fix: Train a decoder to predict next_state, measure real MSE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass

from psa.neuron_vectorized import VectorizedPSALayer, VectorizedPSANetwork


@dataclass
class TrajectoryStats:
    """Statistics for trajectory evaluation."""
    name: str
    mean_mse: float
    std_mse: float
    mean_surprise: float
    num_steps: int


class SimpleDynamics:
    """2D point mass dynamics."""

    def __init__(self, dt: float = 0.1, noise_std: float = 0.01):
        self.dt = dt
        self.noise_std = noise_std
        self.state_dim = 4
        self.action_dim = 2

    def reset(self) -> torch.Tensor:
        x = torch.rand(2) * 2 - 1
        v = torch.rand(2) * 0.2 - 0.1
        return torch.cat([x, v])

    def step(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x, v = state[:2], state[2:]
        v_new = v + action * self.dt
        x_new = x + v_new * self.dt
        noise = torch.randn(4) * self.noise_std
        x_new = torch.clamp(x_new, -2, 2)
        v_new = torch.clamp(v_new, -1, 1)
        return torch.cat([x_new, v_new]) + noise


class ExpertController:
    def __init__(self):
        self.target = torch.zeros(2)

    def __call__(self, state: torch.Tensor) -> torch.Tensor:
        x, v = state[:2], state[2:]
        action = 2.0 * (self.target - x) - 0.5 * v
        return torch.clamp(action, -1, 1)


class RandomController:
    def __call__(self, state: torch.Tensor) -> torch.Tensor:
        return torch.rand(2) * 2 - 1


class NoisyExpertController:
    def __init__(self, noise_std: float = 0.3):
        self.expert = ExpertController()
        self.noise_std = noise_std

    def __call__(self, state: torch.Tensor) -> torch.Tensor:
        action = self.expert(state)
        return torch.clamp(action + torch.randn(2) * self.noise_std, -1, 1)


class OppositeController:
    def __init__(self):
        self.expert = ExpertController()

    def __call__(self, state: torch.Tensor) -> torch.Tensor:
        return -self.expert(state)


def collect_trajectory(dynamics, controller, num_steps: int = 50):
    trajectory = []
    state = dynamics.reset()
    for _ in range(num_steps):
        action = controller(state)
        next_state = dynamics.step(state, action)
        trajectory.append((state.clone(), action.clone(), next_state.clone()))
        state = next_state
    return trajectory


class PSAWorldModel(nn.Module):
    """
    PSA network with a trained decoder for next-state prediction.

    Architecture:
    - PSA encoder: (state, action) → spike representation
    - Decoder: spike representation → predicted next_state

    Training:
    - PSA: local unsupervised learning
    - Decoder: MSE loss on next_state prediction (this uses gradients, but only for decoder)
    """

    def __init__(
        self,
        state_dim: int = 4,
        action_dim: int = 2,
        psa_layers: list = [64, 32],
        decoder_hidden: int = 32,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        # PSA encoder (local learning only)
        self.psa = VectorizedPSANetwork(
            input_dim=state_dim + action_dim,
            layer_sizes=psa_layers,
            prediction_windows=[3] * len(psa_layers),
        )

        # Decoder (trained with gradients to predict next_state)
        top_layer_size = psa_layers[-1]
        self.decoder = nn.Sequential(
            nn.Linear(top_layer_size, decoder_hidden),
            nn.ReLU(),
            nn.Linear(decoder_hidden, state_dim),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        """Forward pass: (state, action) → predicted_next_state."""
        x = torch.cat([state, action], dim=-1)
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # PSA encoding
        spikes, info = self.psa.forward(x.float())

        # Decode to next state
        if spikes.dim() == 1:
            spikes = spikes.unsqueeze(0)
        predicted_next = self.decoder(spikes)

        return predicted_next.squeeze(0), info

    def local_update(self):
        """PSA local plasticity (no gradients)."""
        self.psa.local_update()

    def reset(self):
        """Reset PSA state."""
        self.psa.reset()


def train_world_model(
    trajectories: List[List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]],
    num_psa_epochs: int = 10,
    num_decoder_epochs: int = 50,
    decoder_lr: float = 0.01,
    verbose: bool = True,
):
    """
    Two-phase training:
    1. PSA local learning (unsupervised)
    2. Decoder training (supervised MSE)
    """
    model = PSAWorldModel()

    # Phase 1: PSA local learning
    if verbose:
        print("Phase 1: PSA local learning...")

    for epoch in range(num_psa_epochs):
        for traj in trajectories:
            model.reset()
            for state, action, next_state in traj:
                model.forward(state.float(), action.float())
                model.local_update()

    model.psa.consolidate()

    # Phase 2: Train decoder with gradients
    if verbose:
        print("Phase 2: Decoder training...")

    # Collect all (spikes, next_state) pairs
    all_spikes = []
    all_next_states = []

    for traj in trajectories:
        model.reset()
        for state, action, next_state in traj:
            with torch.no_grad():
                x = torch.cat([state, action]).float().unsqueeze(0)
                spikes, _ = model.psa.forward(x)
                all_spikes.append(spikes.squeeze(0))
                all_next_states.append(next_state.float())

    spikes_tensor = torch.stack(all_spikes)
    next_states_tensor = torch.stack(all_next_states)

    # Train decoder
    optimizer = torch.optim.Adam(model.decoder.parameters(), lr=decoder_lr)

    for epoch in range(num_decoder_epochs):
        optimizer.zero_grad()
        predicted = model.decoder(spikes_tensor)
        loss = F.mse_loss(predicted, next_states_tensor)
        loss.backward()
        optimizer.step()

        if verbose and (epoch + 1) % 10 == 0:
            print(f"  Decoder epoch {epoch+1}/{num_decoder_epochs}: MSE = {loss.item():.6f}")

    return model


def evaluate_model(
    model: PSAWorldModel,
    trajectories: List[List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]],
    controller_name: str,
) -> TrajectoryStats:
    """Evaluate actual prediction MSE on trajectories."""
    mses = []
    surprises = []

    for traj in trajectories:
        model.reset()
        for state, action, next_state in traj:
            with torch.no_grad():
                predicted_next, info = model.forward(state.float(), action.float())

            mse = F.mse_loss(predicted_next, next_state.float()).item()
            mses.append(mse)
            surprises.append(info['surprise'])

    return TrajectoryStats(
        name=controller_name,
        mean_mse=np.mean(mses),
        std_mse=np.std(mses),
        mean_surprise=np.mean(surprises),
        num_steps=len(mses),
    )


def run_test(
    num_train: int = 50,
    num_test: int = 20,
    traj_len: int = 50,
):
    print("=" * 70)
    print("CONTROLLER SWAP TEST V2 - Real Prediction Error")
    print("=" * 70)
    print()

    dynamics = SimpleDynamics()

    controllers = {
        'expert': ExpertController(),
        'random': RandomController(),
        'noisy_expert': NoisyExpertController(noise_std=0.3),
        'noisy_expert_high': NoisyExpertController(noise_std=0.5),
        'opposite': OppositeController(),
    }

    # Collect training data (expert only)
    print("Collecting expert training data...")
    train_trajs = [collect_trajectory(dynamics, controllers['expert'], traj_len) for _ in range(num_train)]

    # Train model
    print()
    model = train_world_model(train_trajs, num_psa_epochs=10, num_decoder_epochs=50)

    # Collect test data
    print("\nCollecting test data for each controller...")
    test_trajs = {name: [collect_trajectory(dynamics, ctrl, traj_len) for _ in range(num_test)]
                  for name, ctrl in controllers.items()}

    # Evaluate
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\n{'Controller':<20} {'Mean MSE':>12} {'Std MSE':>12} {'Ratio':>10}")
    print("-" * 60)

    results = {}
    baseline_mse = None

    for name, trajs in test_trajs.items():
        stats = evaluate_model(model, trajs, name)
        results[name] = stats

        if name == 'expert':
            baseline_mse = stats.mean_mse

        ratio = stats.mean_mse / baseline_mse if baseline_mse else 1.0
        print(f"{name:<20} {stats.mean_mse:>12.6f} {stats.std_mse:>12.6f} {ratio:>10.2f}x")

    # Interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    random_ratio = results['random'].mean_mse / baseline_mse

    if random_ratio < 2.0:
        print(f"✓ Random ratio = {random_ratio:.2f}x → PSA learned transferable dynamics")
    elif random_ratio < 5.0:
        print(f"⚠ Random ratio = {random_ratio:.2f}x → Partial dynamics learning")
    else:
        print(f"✗ Random ratio = {random_ratio:.2f}x → Memorized expert trajectories")

    # Compare to baseline: linear decoder on raw (state, action)
    print("\n" + "=" * 70)
    print("BASELINE: Linear decoder on raw (state, action)")
    print("=" * 70)

    # Train linear baseline
    all_inputs = []
    all_next = []
    for traj in train_trajs:
        for state, action, next_state in traj:
            all_inputs.append(torch.cat([state, action]).float())
            all_next.append(next_state.float())

    inputs = torch.stack(all_inputs)
    targets = torch.stack(all_next)

    linear = nn.Linear(6, 4)
    optimizer = torch.optim.Adam(linear.parameters(), lr=0.01)

    for _ in range(100):
        optimizer.zero_grad()
        loss = F.mse_loss(linear(inputs), targets)
        loss.backward()
        optimizer.step()

    print(f"Linear baseline training MSE: {loss.item():.6f}")

    # Evaluate linear baseline
    print(f"\n{'Controller':<20} {'Linear MSE':>12} {'PSA MSE':>12} {'PSA Better?':>12}")
    print("-" * 60)

    for name, trajs in test_trajs.items():
        linear_mses = []
        for traj in trajs:
            for state, action, next_state in traj:
                with torch.no_grad():
                    pred = linear(torch.cat([state, action]).float())
                    mse = F.mse_loss(pred, next_state.float()).item()
                    linear_mses.append(mse)

        linear_mean = np.mean(linear_mses)
        psa_mean = results[name].mean_mse
        better = "✓ Yes" if psa_mean < linear_mean else "✗ No"
        print(f"{name:<20} {linear_mean:>12.6f} {psa_mean:>12.6f} {better:>12}")

    return results


if __name__ == "__main__":
    results = run_test(num_train=50, num_test=20, traj_len=50)
