"""
Test Predictive PSA - Does predicting next observation work better?

Compare:
1. Original PSA (predicts neighbor spikes)
2. Predictive PSA (predicts next observation)
3. Linear baseline

All evaluated with controller swap test.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass

from psa.predictive_psa import PredictivePSAWorldModel, PredictivePSANetwork
from psa.neuron_vectorized import VectorizedPSANetwork


@dataclass
class Results:
    name: str
    expert_mse: float
    random_mse: float
    noisy_mse: float
    opposite_mse: float
    ratio_random: float
    ratio_opposite: float


class SimpleDynamics:
    """Simple 2D dynamics for testing."""

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
        x_new = torch.clamp(x_new, -2, 2)
        v_new = torch.clamp(v_new, -1, 1)
        return torch.cat([x_new, v_new]) + torch.randn(4) * self.noise_std


class ExpertController:
    def __call__(self, state: torch.Tensor) -> torch.Tensor:
        x, v = state[:2], state[2:]
        action = 2.0 * (-x) - 0.5 * v
        return torch.clamp(action, -1, 1)


class RandomController:
    def __call__(self, state: torch.Tensor) -> torch.Tensor:
        return torch.rand(2) * 2 - 1


class NoisyExpertController:
    def __init__(self, noise_std: float = 0.3):
        self.expert = ExpertController()
        self.noise_std = noise_std

    def __call__(self, state: torch.Tensor) -> torch.Tensor:
        return torch.clamp(self.expert(state) + torch.randn(2) * self.noise_std, -1, 1)


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


def train_predictive_psa(
    trajectories: List,
    num_epochs: int = 20,
    verbose: bool = True,
):
    """Train Predictive PSA network with pure local learning.

    Direct obs → PSA → predicted_next_obs (no separate encoder).
    """

    # Direct PSA on raw observations
    psa = PredictivePSANetwork(
        obs_dim=4,
        action_dim=2,
        layer_sizes=[64, 32],
    )

    # Simple linear decoder from PSA layer prediction to observation
    # (trained with gradient descent - this is the only non-local part)
    decoder = nn.Linear(4, 4)  # First layer predicts obs_dim latent
    optimizer = torch.optim.Adam(decoder.parameters(), lr=0.01, weight_decay=0.01)

    if verbose:
        print("Training Predictive PSA (local PSA + linear decoder)...")

    for epoch in range(num_epochs):
        total_error = 0.0
        total_steps = 0

        for traj in trajectories:
            psa.reset()

            for state, action, next_state in traj:
                # Forward pass
                spikes, predictions, info = psa(state, action)

                # predictions[0] is first layer's prediction of next obs
                latent_pred = predictions[0]

                # Decode to observation space
                predicted_next = decoder(latent_pred)

                # Train decoder
                loss = F.mse_loss(predicted_next, next_state.float())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Local update for PSA (the key part)
                psa.local_update(next_state)

                total_error += loss.item()
                total_steps += 1

        if verbose and (epoch + 1) % 5 == 0:
            avg_error = total_error / total_steps
            print(f"  Epoch {epoch+1}/{num_epochs}: MSE = {avg_error:.6f}")

    psa.consolidate()
    return psa, decoder


def evaluate_predictive_psa(
    psa: PredictivePSANetwork,
    decoder: nn.Linear,
    trajectories: List,
) -> float:
    """Evaluate prediction MSE."""
    total_mse = 0.0
    total_steps = 0

    for traj in trajectories:
        psa.reset()

        for state, action, next_state in traj:
            with torch.no_grad():
                spikes, predictions, info = psa(state, action)
                latent_pred = predictions[0]
                predicted_next = decoder(latent_pred)
                mse = F.mse_loss(predicted_next, next_state.float()).item()
                total_mse += mse
                total_steps += 1

    return total_mse / total_steps


def train_original_psa(
    trajectories: List,
    num_epochs: int = 20,
    verbose: bool = True,
):
    """Train original PSA (neighbor prediction) + decoder."""

    psa = VectorizedPSANetwork(
        input_dim=6,  # state + action
        layer_sizes=[64, 32],
        prediction_windows=[3, 5],
    )

    decoder = nn.Sequential(
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 4),
    )

    if verbose:
        print("Training Original PSA...")

    # Phase 1: PSA local learning
    for epoch in range(num_epochs):
        for traj in trajectories:
            psa.reset()
            for state, action, next_state in traj:
                x = torch.cat([state, action]).float()
                psa.forward(x)
                psa.local_update()

    psa.consolidate()

    # Phase 2: Train decoder
    all_spikes = []
    all_next = []

    for traj in trajectories:
        psa.reset()
        for state, action, next_state in traj:
            with torch.no_grad():
                x = torch.cat([state, action]).float()
                spikes, _ = psa.forward(x)
                all_spikes.append(spikes.squeeze(0) if spikes.dim() > 1 else spikes)
                all_next.append(next_state.float())

    spikes_tensor = torch.stack(all_spikes)
    next_tensor = torch.stack(all_next)

    optimizer = torch.optim.Adam(decoder.parameters(), lr=0.01)
    for _ in range(100):
        optimizer.zero_grad()
        loss = F.mse_loss(decoder(spikes_tensor), next_tensor)
        loss.backward()
        optimizer.step()

    if verbose:
        print(f"  Decoder trained: MSE = {loss.item():.6f}")

    return psa, decoder


def evaluate_original_psa(psa, decoder, trajectories: List) -> float:
    """Evaluate original PSA + decoder."""
    total_mse = 0.0
    total_steps = 0

    for traj in trajectories:
        psa.reset()
        for state, action, next_state in traj:
            with torch.no_grad():
                x = torch.cat([state, action]).float()
                spikes, _ = psa.forward(x)
                if spikes.dim() > 1:
                    spikes = spikes.squeeze(0)
                predicted = decoder(spikes)
                mse = F.mse_loss(predicted, next_state.float()).item()
                total_mse += mse
                total_steps += 1

    return total_mse / total_steps


def train_linear_baseline(trajectories: List) -> nn.Linear:
    """Train linear baseline."""
    all_inputs = []
    all_next = []

    for traj in trajectories:
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

    return linear


def evaluate_linear(linear, trajectories: List) -> float:
    """Evaluate linear baseline."""
    total_mse = 0.0
    total_steps = 0

    for traj in trajectories:
        for state, action, next_state in traj:
            with torch.no_grad():
                x = torch.cat([state, action]).float()
                predicted = linear(x)
                mse = F.mse_loss(predicted, next_state.float()).item()
                total_mse += mse
                total_steps += 1

    return total_mse / total_steps


def run_comparison():
    """Compare all methods with controller swap test."""

    print("=" * 70)
    print("PREDICTIVE PSA vs ORIGINAL PSA vs LINEAR BASELINE")
    print("=" * 70)
    print()

    dynamics = SimpleDynamics()

    controllers = {
        'expert': ExpertController(),
        'random': RandomController(),
        'noisy': NoisyExpertController(noise_std=0.3),
        'opposite': OppositeController(),
    }

    # Collect training data (expert only)
    print("Collecting training data (expert trajectories)...")
    train_trajs = [collect_trajectory(dynamics, controllers['expert'], 50) for _ in range(50)]

    # Collect test data (all controllers)
    print("Collecting test data...")
    test_trajs = {
        name: [collect_trajectory(dynamics, ctrl, 50) for _ in range(20)]
        for name, ctrl in controllers.items()
    }

    results = {}

    # 1. Predictive PSA
    print("\n" + "-" * 50)
    pred_psa, pred_decoder = train_predictive_psa(train_trajs, num_epochs=20)

    pred_results = {}
    for name, trajs in test_trajs.items():
        pred_results[name] = evaluate_predictive_psa(pred_psa, pred_decoder, trajs)

    results['Predictive PSA'] = Results(
        name='Predictive PSA',
        expert_mse=pred_results['expert'],
        random_mse=pred_results['random'],
        noisy_mse=pred_results['noisy'],
        opposite_mse=pred_results['opposite'],
        ratio_random=pred_results['random'] / pred_results['expert'],
        ratio_opposite=pred_results['opposite'] / pred_results['expert'],
    )

    # 2. Original PSA
    print("\n" + "-" * 50)
    orig_psa, decoder = train_original_psa(train_trajs, num_epochs=20)

    orig_results = {}
    for name, trajs in test_trajs.items():
        orig_results[name] = evaluate_original_psa(orig_psa, decoder, trajs)

    results['Original PSA'] = Results(
        name='Original PSA',
        expert_mse=orig_results['expert'],
        random_mse=orig_results['random'],
        noisy_mse=orig_results['noisy'],
        opposite_mse=orig_results['opposite'],
        ratio_random=orig_results['random'] / orig_results['expert'],
        ratio_opposite=orig_results['opposite'] / orig_results['expert'],
    )

    # 3. Linear baseline
    print("\n" + "-" * 50)
    print("Training Linear baseline...")
    linear = train_linear_baseline(train_trajs)

    linear_results = {}
    for name, trajs in test_trajs.items():
        linear_results[name] = evaluate_linear(linear, trajs)

    results['Linear'] = Results(
        name='Linear',
        expert_mse=linear_results['expert'],
        random_mse=linear_results['random'],
        noisy_mse=linear_results['noisy'],
        opposite_mse=linear_results['opposite'],
        ratio_random=linear_results['random'] / linear_results['expert'],
        ratio_opposite=linear_results['opposite'] / linear_results['expert'],
    )

    # Print results table
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()
    print(f"{'Method':<20} {'Expert':>10} {'Random':>10} {'Noisy':>10} {'Opposite':>10} {'R-Ratio':>10} {'O-Ratio':>10}")
    print("-" * 80)

    for name, r in results.items():
        print(f"{name:<20} {r.expert_mse:>10.6f} {r.random_mse:>10.6f} "
              f"{r.noisy_mse:>10.6f} {r.opposite_mse:>10.6f} "
              f"{r.ratio_random:>10.2f}x {r.ratio_opposite:>10.2f}x")

    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    # Best on expert?
    expert_mses = [(name, r.expert_mse) for name, r in results.items()]
    best_expert = min(expert_mses, key=lambda x: x[1])
    print(f"\nBest expert prediction: {best_expert[0]} (MSE={best_expert[1]:.6f})")

    # Best generalization (lowest ratio)?
    ratios = [(name, r.ratio_random) for name, r in results.items()]
    best_ratio = min(ratios, key=lambda x: x[1])
    print(f"Best generalization (random): {best_ratio[0]} (ratio={best_ratio[1]:.2f}x)")

    # Does Predictive PSA beat Original PSA?
    pred = results['Predictive PSA']
    orig = results['Original PSA']

    if pred.ratio_random < orig.ratio_random:
        print(f"\n✓ Predictive PSA generalizes better than Original PSA")
        print(f"  Random ratio: {pred.ratio_random:.2f}x vs {orig.ratio_random:.2f}x")
    else:
        print(f"\n✗ Original PSA generalizes as well or better")
        print(f"  Random ratio: {pred.ratio_random:.2f}x vs {orig.ratio_random:.2f}x")

    # Does PSA beat Linear?
    lin = results['Linear']
    if pred.expert_mse < lin.expert_mse:
        print(f"\n✓ Predictive PSA beats Linear on expert")
    else:
        print(f"\n✗ Linear beats Predictive PSA on expert")

    print()
    return results


if __name__ == "__main__":
    results = run_comparison()
