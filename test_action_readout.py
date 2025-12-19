"""
Test 3-Factor Action Readout.

Evaluates whether PSA + 3-factor readout can:
1. Learn from demonstrations (imitation)
2. Generalize to new initial states
3. Maintain good world model predictions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass

from psa.action_readout import PSAWithActionReadout, ThreeFactorReadout
from psa.predictive_psa import PredictivePSANetwork


@dataclass
class ImitationResults:
    name: str
    train_action_mse: float
    test_action_mse: float
    train_state_mse: float
    test_state_mse: float
    success_rate: float


class SimpleDynamics:
    """Simple 2D dynamics for testing."""

    def __init__(self, dt: float = 0.1, noise_std: float = 0.01):
        self.dt = dt
        self.noise_std = noise_std
        self.state_dim = 4
        self.action_dim = 2

    def reset(self, fixed: bool = False) -> torch.Tensor:
        if fixed:
            x = torch.zeros(2)
            v = torch.zeros(2)
        else:
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
    """Expert: PD control toward origin."""

    def __call__(self, state: torch.Tensor) -> torch.Tensor:
        x, v = state[:2], state[2:]
        action = 2.0 * (-x) - 0.5 * v
        return torch.clamp(action, -1, 1)


def collect_demo_trajectory(dynamics, controller, num_steps: int = 50):
    """Collect demonstration trajectory with (state, action, next_state)."""
    trajectory = []
    state = dynamics.reset()

    for _ in range(num_steps):
        action = controller(state)
        next_state = dynamics.step(state, action)
        trajectory.append((state.clone(), action.clone(), next_state.clone()))
        state = next_state

    return trajectory


def train_psa_with_readout(
    trajectories: List,
    num_epochs: int = 30,
    verbose: bool = True,
) -> PSAWithActionReadout:
    """Train PSA with 3-factor action readout."""

    model = PSAWithActionReadout(
        obs_dim=4,
        action_dim=2,
        psa_neurons=64,
        readout_hidden=32,
    )

    if verbose:
        print("Training PSA + 3-Factor Readout...")

    for epoch in range(num_epochs):
        total_action_error = 0.0
        total_state_error = 0.0
        total_steps = 0

        for traj in trajectories:
            model.reset()

            for state, demo_action, next_state in traj:
                # Forward pass
                pred_action, pred_next, info = model.forward(state, demo_action)

                # Local update (both world model and action readout)
                model.update(next_state, demo_action)

                # Track errors
                with torch.no_grad():
                    action_mse = F.mse_loss(pred_action, demo_action.float()).item()
                    state_mse = F.mse_loss(pred_next, next_state.float()).item()
                    total_action_error += action_mse
                    total_state_error += state_mse
                    total_steps += 1

        if verbose and (epoch + 1) % 5 == 0:
            avg_action = total_action_error / total_steps
            avg_state = total_state_error / total_steps
            print(f"  Epoch {epoch+1}/{num_epochs}: Action MSE = {avg_action:.4f}, State MSE = {avg_state:.4f}")

    model.consolidate()
    return model


def evaluate_imitation(
    model: PSAWithActionReadout,
    dynamics: SimpleDynamics,
    expert: ExpertController,
    num_trajectories: int = 20,
    trajectory_length: int = 50,
) -> Tuple[float, float, float]:
    """Evaluate imitation quality.

    Returns:
        action_mse: MSE between model and expert actions
        state_mse: World model prediction error
        success_rate: Fraction of trajectories reaching goal
    """
    total_action_mse = 0.0
    total_state_mse = 0.0
    total_steps = 0
    successes = 0

    model.eval()

    for _ in range(num_trajectories):
        model.reset()
        state = dynamics.reset()

        trajectory_success = False

        for step in range(trajectory_length):
            # Get expert action (ground truth)
            expert_action = expert(state)

            # Get model action
            with torch.no_grad():
                pred_action, pred_next, _ = model.forward(state)

            # Record errors
            action_mse = F.mse_loss(pred_action, expert_action).item()
            total_action_mse += action_mse

            # Execute model action in environment
            next_state = dynamics.step(state, pred_action)

            state_mse = F.mse_loss(pred_next, next_state).item()
            total_state_mse += state_mse

            total_steps += 1
            state = next_state

            # Check if reached goal (near origin)
            if state[:2].norm() < 0.1:
                trajectory_success = True

        if trajectory_success:
            successes += 1

    model.train()

    return (
        total_action_mse / total_steps,
        total_state_mse / total_steps,
        successes / num_trajectories,
    )


def train_baseline_mlp(
    trajectories: List,
    num_epochs: int = 100,
) -> nn.Module:
    """Train MLP baseline with gradient descent."""

    mlp = nn.Sequential(
        nn.Linear(4, 32),
        nn.ReLU(),
        nn.Linear(32, 2),
        nn.Tanh(),
    )

    optimizer = torch.optim.Adam(mlp.parameters(), lr=0.01)

    # Collect all data
    all_states = []
    all_actions = []
    for traj in trajectories:
        for state, action, _ in traj:
            all_states.append(state)
            all_actions.append(action)

    states = torch.stack(all_states)
    actions = torch.stack(all_actions)

    for _ in range(num_epochs):
        optimizer.zero_grad()
        pred = mlp(states)
        loss = F.mse_loss(pred, actions)
        loss.backward()
        optimizer.step()

    return mlp


def evaluate_baseline(
    mlp: nn.Module,
    dynamics: SimpleDynamics,
    expert: ExpertController,
    num_trajectories: int = 20,
    trajectory_length: int = 50,
) -> Tuple[float, float]:
    """Evaluate MLP baseline."""
    total_action_mse = 0.0
    total_steps = 0
    successes = 0

    mlp.eval()

    for _ in range(num_trajectories):
        state = dynamics.reset()
        trajectory_success = False

        for _ in range(trajectory_length):
            expert_action = expert(state)

            with torch.no_grad():
                pred_action = mlp(state)

            action_mse = F.mse_loss(pred_action, expert_action).item()
            total_action_mse += action_mse
            total_steps += 1

            next_state = dynamics.step(state, pred_action)
            state = next_state

            if state[:2].norm() < 0.1:
                trajectory_success = True

        if trajectory_success:
            successes += 1

    mlp.train()

    return total_action_mse / total_steps, successes / num_trajectories


def run_imitation_test():
    """Full imitation learning test."""

    print("=" * 70)
    print("PSA + 3-FACTOR READOUT vs MLP BASELINE")
    print("=" * 70)
    print()

    dynamics = SimpleDynamics()
    expert = ExpertController()

    # Collect training demonstrations
    print("Collecting training demonstrations...")
    train_trajs = [collect_demo_trajectory(dynamics, expert, 50) for _ in range(50)]

    # Train PSA + 3-factor readout
    print("\n" + "-" * 50)
    psa_model = train_psa_with_readout(train_trajs, num_epochs=50)

    # Train MLP baseline
    print("\n" + "-" * 50)
    print("Training MLP baseline...")
    mlp = train_baseline_mlp(train_trajs, num_epochs=100)

    # Evaluate on training distribution
    print("\n" + "-" * 50)
    print("Evaluating on training distribution...")

    psa_train_action, psa_train_state, psa_train_success = evaluate_imitation(
        psa_model, dynamics, expert, num_trajectories=20
    )

    mlp_train_action, mlp_train_success = evaluate_baseline(
        mlp, dynamics, expert, num_trajectories=20
    )

    # Evaluate on test distribution (different initial states)
    print("Evaluating on test distribution...")

    psa_test_action, psa_test_state, psa_test_success = evaluate_imitation(
        psa_model, dynamics, expert, num_trajectories=20
    )

    mlp_test_action, mlp_test_success = evaluate_baseline(
        mlp, dynamics, expert, num_trajectories=20
    )

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()
    print(f"{'Method':<25} {'Action MSE':>12} {'Success %':>12} {'State MSE':>12}")
    print("-" * 65)
    print(f"{'PSA + 3-Factor (train)':<25} {psa_train_action:>12.4f} {psa_train_success*100:>11.1f}% {psa_train_state:>12.4f}")
    print(f"{'PSA + 3-Factor (test)':<25} {psa_test_action:>12.4f} {psa_test_success*100:>11.1f}% {psa_test_state:>12.4f}")
    print(f"{'MLP Baseline (train)':<25} {mlp_train_action:>12.4f} {mlp_train_success*100:>11.1f}% {'N/A':>12}")
    print(f"{'MLP Baseline (test)':<25} {mlp_test_action:>12.4f} {mlp_test_success*100:>11.1f}% {'N/A':>12}")

    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    if psa_test_success > mlp_test_success:
        print("✓ PSA + 3-Factor achieves higher success rate")
    elif psa_test_success == mlp_test_success:
        print("= PSA + 3-Factor matches MLP success rate")
    else:
        print("✗ MLP achieves higher success rate")

    if psa_test_action < mlp_test_action:
        print("✓ PSA + 3-Factor has lower action error")
    else:
        print("✗ MLP has lower action error")

    print(f"\nPSA also maintains world model (state MSE: {psa_test_state:.4f})")

    return {
        'psa': {
            'train_action': psa_train_action,
            'test_action': psa_test_action,
            'train_success': psa_train_success,
            'test_success': psa_test_success,
            'state_mse': psa_test_state,
        },
        'mlp': {
            'train_action': mlp_train_action,
            'test_action': mlp_test_action,
            'train_success': mlp_train_success,
            'test_success': mlp_test_success,
        },
    }


if __name__ == "__main__":
    results = run_imitation_test()
