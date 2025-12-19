"""
Controller Swap Test for PSA World Model Validation.

Key question: Did PSA learn dynamics or just memorize demo trajectories?

Test protocol:
1. Train PSA on expert demo trajectories
2. Evaluate prediction error on:
   - Held-out expert demos (baseline)
   - Random policy trajectories
   - Noisy expert trajectories
   - Different controller trajectories

If error stays similar across controllers → learned dynamics
If error spikes on non-expert → memorized demos
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import time

from psa.neuron_vectorized import VectorizedPSALayer, VectorizedPSANetwork


@dataclass
class TrajectoryStats:
    """Statistics for a trajectory set."""
    name: str
    mean_prediction_error: float
    std_prediction_error: float
    mean_surprise: float
    num_trajectories: int
    num_steps: int


class SimpleDynamics:
    """
    Simple 2D dynamics for testing.

    State: [x, y, vx, vy]
    Action: [ax, ay] (acceleration)

    Dynamics:
        v' = v + a * dt
        x' = x + v' * dt
    """

    def __init__(self, dt: float = 0.1, noise_std: float = 0.01):
        self.dt = dt
        self.noise_std = noise_std
        self.state_dim = 4
        self.action_dim = 2

    def reset(self) -> torch.Tensor:
        """Reset to random initial state."""
        x = torch.rand(2) * 2 - 1  # Position in [-1, 1]
        v = torch.rand(2) * 0.2 - 0.1  # Velocity in [-0.1, 0.1]
        return torch.cat([x, v])

    def step(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Apply dynamics."""
        x, v = state[:2], state[2:]
        a = action

        # Physics
        v_new = v + a * self.dt
        x_new = x + v_new * self.dt

        # Add noise
        noise = torch.randn(4) * self.noise_std

        # Clamp to bounds
        x_new = torch.clamp(x_new, -2, 2)
        v_new = torch.clamp(v_new, -1, 1)

        return torch.cat([x_new, v_new]) + noise


class ExpertController:
    """Expert: moves toward target with PD control."""

    def __init__(self, target: torch.Tensor = None, kp: float = 2.0, kd: float = 0.5):
        self.target = target if target is not None else torch.zeros(2)
        self.kp = kp
        self.kd = kd

    def __call__(self, state: torch.Tensor) -> torch.Tensor:
        x, v = state[:2], state[2:]
        error = self.target - x
        action = self.kp * error - self.kd * v
        return torch.clamp(action, -1, 1)


class RandomController:
    """Random: uniform random actions."""

    def __call__(self, state: torch.Tensor) -> torch.Tensor:
        return torch.rand(2) * 2 - 1


class NoisyExpertController:
    """Noisy expert: expert + Gaussian noise."""

    def __init__(self, noise_std: float = 0.3):
        self.expert = ExpertController()
        self.noise_std = noise_std

    def __call__(self, state: torch.Tensor) -> torch.Tensor:
        action = self.expert(state)
        noise = torch.randn(2) * self.noise_std
        return torch.clamp(action + noise, -1, 1)


class OppositeController:
    """Opposite: moves away from target."""

    def __init__(self):
        self.expert = ExpertController()

    def __call__(self, state: torch.Tensor) -> torch.Tensor:
        return -self.expert(state)


class OscillatingController:
    """Oscillating: sinusoidal actions."""

    def __init__(self, freq: float = 1.0):
        self.freq = freq
        self.t = 0

    def __call__(self, state: torch.Tensor) -> torch.Tensor:
        self.t += 0.1
        return torch.tensor([
            np.sin(self.freq * self.t),
            np.cos(self.freq * self.t)
        ])


def collect_trajectory(
    dynamics: SimpleDynamics,
    controller,
    num_steps: int = 50,
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Collect a trajectory.

    Returns list of (state, action, next_state) tuples.
    """
    trajectory = []
    state = dynamics.reset()

    for _ in range(num_steps):
        action = controller(state)
        next_state = dynamics.step(state, action)
        trajectory.append((state.clone(), action.clone(), next_state.clone()))
        state = next_state

    return trajectory


def train_psa_world_model(
    trajectories: List[List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]],
    num_epochs: int = 10,
    verbose: bool = True,
) -> VectorizedPSANetwork:
    """
    Train PSA world model on trajectories.

    The model learns to predict next state given (state, action).
    """
    state_dim = 4
    action_dim = 2
    input_dim = state_dim + action_dim

    # Create network
    psa = VectorizedPSANetwork(
        input_dim=input_dim,
        layer_sizes=[64, 32],
        prediction_windows=[3, 5],
        neighbor_radius=3,
    )

    if verbose:
        print(f"Training PSA world model on {len(trajectories)} trajectories...")

    for epoch in range(num_epochs):
        total_error = 0.0
        total_steps = 0

        for traj in trajectories:
            psa.reset()

            for state, action, next_state in traj:
                # Input: state + action
                x = torch.cat([state, action]).float()

                # Forward pass
                output, info = psa.forward(x)

                # Local update (unsupervised prediction)
                psa.local_update()

                # Track error (for monitoring only)
                total_error += info['surprise']
                total_steps += 1

        if verbose and (epoch + 1) % 2 == 0:
            avg_error = total_error / total_steps
            print(f"  Epoch {epoch+1}/{num_epochs}: avg_surprise = {avg_error:.4f}")

    # Consolidate learned patterns
    psa.consolidate()

    return psa


def evaluate_prediction_error(
    psa: VectorizedPSANetwork,
    trajectories: List[List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]],
    controller_name: str,
) -> TrajectoryStats:
    """
    Evaluate prediction error on trajectories.

    This measures how well PSA predicts next states for this controller.
    """
    errors = []
    surprises = []
    total_steps = 0

    for traj in trajectories:
        psa.reset()

        for state, action, next_state in traj:
            x = torch.cat([state, action]).float()

            # Get prediction
            output, info = psa.forward(x)

            # Record surprise (proxy for prediction quality)
            surprises.append(info['surprise'])

            # Compute actual prediction error
            # (We'd need a decoder to predict next_state from output)
            # For now, use internal surprise as proxy
            errors.append(info['surprise'])
            total_steps += 1

    return TrajectoryStats(
        name=controller_name,
        mean_prediction_error=np.mean(errors),
        std_prediction_error=np.std(errors),
        mean_surprise=np.mean(surprises),
        num_trajectories=len(trajectories),
        num_steps=total_steps,
    )


def run_controller_swap_test(
    num_train_trajectories: int = 50,
    num_test_trajectories: int = 20,
    trajectory_length: int = 50,
    num_epochs: int = 10,
):
    """
    Full controller swap test.
    """
    print("=" * 60)
    print("CONTROLLER SWAP TEST")
    print("=" * 60)
    print()

    # Setup
    dynamics = SimpleDynamics()

    controllers = {
        'expert': ExpertController(),
        'random': RandomController(),
        'noisy_expert': NoisyExpertController(noise_std=0.3),
        'opposite': OppositeController(),
        'oscillating': OscillatingController(),
    }

    # Collect training data (expert only)
    print("Collecting expert training trajectories...")
    train_trajectories = [
        collect_trajectory(dynamics, controllers['expert'], trajectory_length)
        for _ in range(num_train_trajectories)
    ]

    # Train PSA
    psa = train_psa_world_model(train_trajectories, num_epochs=num_epochs)

    # Collect test data for each controller
    print("\nCollecting test trajectories for each controller...")
    test_trajectories = {}
    for name, controller in controllers.items():
        if name == 'oscillating':
            controller = OscillatingController()  # Reset time
        test_trajectories[name] = [
            collect_trajectory(dynamics, controller, trajectory_length)
            for _ in range(num_test_trajectories)
        ]

    # Evaluate on each controller
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print()
    print(f"{'Controller':<20} {'Mean Error':>12} {'Std Error':>12} {'Surprise':>12} {'Ratio':>10}")
    print("-" * 70)

    results = {}
    baseline_error = None

    for name, trajs in test_trajectories.items():
        stats = evaluate_prediction_error(psa, trajs, name)
        results[name] = stats

        if name == 'expert':
            baseline_error = stats.mean_prediction_error

        ratio = stats.mean_prediction_error / baseline_error if baseline_error else 1.0

        print(f"{name:<20} {stats.mean_prediction_error:>12.4f} {stats.std_prediction_error:>12.4f} "
              f"{stats.mean_surprise:>12.4f} {ratio:>10.2f}x")

    # Interpretation
    print()
    print("=" * 60)
    print("INTERPRETATION")
    print("=" * 60)

    random_ratio = results['random'].mean_prediction_error / baseline_error
    noisy_ratio = results['noisy_expert'].mean_prediction_error / baseline_error

    if random_ratio < 2.0:
        print("✓ Random policy error < 2x expert → PSA likely learned dynamics")
    elif random_ratio < 5.0:
        print("⚠ Random policy error 2-5x expert → PSA partially learned dynamics")
    else:
        print("✗ Random policy error > 5x expert → PSA memorized expert trajectories")

    if noisy_ratio < 1.5:
        print("✓ Noisy expert error < 1.5x expert → Robust to action noise")
    else:
        print("⚠ Noisy expert error > 1.5x expert → Sensitive to action distribution")

    return results


def run_ablation_study():
    """
    Ablation: How does network size affect generalization?
    """
    print("\n" + "=" * 60)
    print("ABLATION: Network Size vs Generalization")
    print("=" * 60)

    dynamics = SimpleDynamics()
    expert = ExpertController()
    random_ctrl = RandomController()

    # Collect data
    train_trajs = [collect_trajectory(dynamics, expert, 50) for _ in range(50)]
    test_expert = [collect_trajectory(dynamics, expert, 50) for _ in range(10)]
    test_random = [collect_trajectory(dynamics, random_ctrl, 50) for _ in range(10)]

    layer_configs = [
        ([16], "Tiny (16)"),
        ([32, 16], "Small (32-16)"),
        ([64, 32], "Medium (64-32)"),
        ([128, 64, 32], "Large (128-64-32)"),
    ]

    print(f"\n{'Config':<25} {'Expert Err':>12} {'Random Err':>12} {'Ratio':>10}")
    print("-" * 60)

    for layers, name in layer_configs:
        # Train
        psa = VectorizedPSANetwork(
            input_dim=6,
            layer_sizes=layers,
            prediction_windows=[3] * len(layers),
        )

        for _ in range(10):
            for traj in train_trajs:
                psa.reset()
                for s, a, ns in traj:
                    x = torch.cat([s, a]).float()
                    psa.forward(x)
                    psa.local_update()

        psa.consolidate()

        # Evaluate
        expert_stats = evaluate_prediction_error(psa, test_expert, "expert")
        random_stats = evaluate_prediction_error(psa, test_random, "random")

        ratio = random_stats.mean_prediction_error / (expert_stats.mean_prediction_error + 1e-6)

        print(f"{name:<25} {expert_stats.mean_prediction_error:>12.4f} "
              f"{random_stats.mean_prediction_error:>12.4f} {ratio:>10.2f}x")


if __name__ == "__main__":
    # Run main test
    results = run_controller_swap_test(
        num_train_trajectories=50,
        num_test_trajectories=20,
        trajectory_length=50,
        num_epochs=10,
    )

    # Run ablation
    run_ablation_study()

    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)
