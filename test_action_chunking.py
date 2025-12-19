"""
Test Action Chunking for Smoother Control.

Key success criteria (per user specification):
1. Lower action variance / jerk
2. Better grasp stability (simulated via trajectory smoothness)
3. Higher success on 'close to contact' segments

Test protocol:
1. Train both single-step and chunked readout on same demos
2. Compare action MSE, success rate, and smoothness metrics
3. Evaluate especially on "precision" segments (near goal)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass

from psa.action_readout import PSAWithActionReadout, PSAWithChunkedReadout


@dataclass
class ChunkingResults:
    name: str
    action_mse: float
    success_rate: float
    action_jerk: float  # Lower is better (smoother)
    precision_success: float  # Success rate near goal


class PrecisionDynamics:
    """
    Environment that tests precision control.

    Has two phases:
    1. Approach phase: Move toward target region
    2. Precision phase: Fine positioning within target region

    Chunking should help especially in precision phase.
    """

    def __init__(self, dt: float = 0.1, noise_std: float = 0.01):
        self.dt = dt
        self.noise_std = noise_std
        self.state_dim = 4  # [x, y, vx, vy]
        self.action_dim = 2
        self.target = torch.tensor([0.8, 0.0])  # Target position
        self.precision_radius = 0.2  # "Close to contact" region

    def reset(self) -> torch.Tensor:
        # Start from random position, away from target
        angle = torch.rand(1).item() * 2 * np.pi
        radius = 0.5 + torch.rand(1).item() * 0.5  # [0.5, 1.0] away from origin
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        v = torch.zeros(2)
        pos = torch.tensor([x, y], dtype=torch.float32)
        return torch.cat([pos, v])

    def step(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x, v = state[:2], state[2:]
        # Smooth dynamics
        v_new = 0.9 * v + 0.1 * action  # Velocity smoothing
        x_new = x + v_new * self.dt
        x_new = torch.clamp(x_new, -2, 2)
        v_new = torch.clamp(v_new, -1, 1)
        return torch.cat([x_new, v_new]) + torch.randn(4) * self.noise_std

    def in_precision_zone(self, state: torch.Tensor) -> bool:
        """Check if state is in precision zone (close to target)."""
        dist = (state[:2] - self.target).norm()
        return dist < self.precision_radius

    def reached_goal(self, state: torch.Tensor) -> bool:
        """Check if successfully reached goal."""
        dist = (state[:2] - self.target).norm()
        speed = state[2:].norm()
        return dist < 0.1 and speed < 0.1


class PrecisionController:
    """Expert controller with smooth approach and precise final positioning."""

    def __init__(self, target: torch.Tensor):
        self.target = target

    def __call__(self, state: torch.Tensor) -> torch.Tensor:
        x, v = state[:2], state[2:]
        error = self.target - x
        dist = error.norm()

        # Adaptive gains: slower near target for precision
        if dist < 0.2:
            # Precision mode: gentle, damped control
            kp, kd = 1.0, 1.0
        else:
            # Approach mode: faster movement
            kp, kd = 2.0, 0.5

        action = kp * error - kd * v
        return torch.clamp(action, -1, 1)


def collect_demos(
    dynamics: PrecisionDynamics,
    controller: PrecisionController,
    num_demos: int = 50,
    demo_length: int = 80,
) -> List[List[Tuple]]:
    """Collect demonstration trajectories."""
    demos = []
    for _ in range(num_demos):
        trajectory = []
        state = dynamics.reset()
        for _ in range(demo_length):
            action = controller(state)
            next_state = dynamics.step(state, action)
            trajectory.append((state.clone(), action.clone(), next_state.clone()))
            state = next_state
        demos.append(trajectory)
    return demos


def compute_jerk(actions: List[torch.Tensor]) -> float:
    """Compute action jerk (rate of change of acceleration proxy)."""
    if len(actions) < 3:
        return 0.0

    jerks = []
    for i in range(2, len(actions)):
        # Second derivative approximation
        jerk = actions[i] - 2 * actions[i-1] + actions[i-2]
        jerks.append(jerk.norm().item())

    return np.mean(jerks)


def train_single_step(
    trajectories: List,
    num_epochs: int = 50,
    verbose: bool = True,
) -> PSAWithActionReadout:
    """Train single-step PSA model."""

    model = PSAWithActionReadout(
        obs_dim=4,
        action_dim=2,
        psa_neurons=64,
        readout_hidden=32,
    )

    if verbose:
        print("Training Single-Step PSA...")

    for epoch in range(num_epochs):
        total_error = 0.0
        total_steps = 0

        for traj in trajectories:
            model.reset()
            for state, demo_action, next_state in traj:
                pred_action, pred_next, _ = model.forward(state, demo_action)
                model.update(next_state, demo_action)

                with torch.no_grad():
                    total_error += F.mse_loss(pred_action, demo_action).item()
                    total_steps += 1

        if verbose and (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{num_epochs}: Action MSE = {total_error/total_steps:.4f}")

    model.consolidate()
    return model


def train_chunked(
    trajectories: List,
    chunk_size: int = 5,
    num_epochs: int = 50,
    verbose: bool = True,
) -> PSAWithChunkedReadout:
    """Train chunked PSA model."""

    model = PSAWithChunkedReadout(
        obs_dim=4,
        action_dim=2,
        chunk_size=chunk_size,
        psa_neurons=64,
        readout_hidden=64,  # Larger for chunk output
    )

    if verbose:
        print(f"Training Chunked PSA (K={chunk_size})...")

    for epoch in range(num_epochs):
        total_error = 0.0
        total_steps = 0

        for traj in trajectories:
            model.reset()

            for i, (state, demo_action, next_state) in enumerate(traj):
                # Get chunk of future demo actions
                demo_chunk = []
                for j in range(chunk_size):
                    if i + j < len(traj):
                        demo_chunk.append(traj[i + j][1])
                    else:
                        demo_chunk.append(traj[-1][1])  # Repeat last action
                demo_chunk = torch.stack(demo_chunk)

                # Forward (receding horizon)
                pred_action, pred_next, info = model.forward(state, receding_horizon=True)

                # Update with chunk target
                model.update(next_state, demo_chunk)

                with torch.no_grad():
                    total_error += F.mse_loss(pred_action, demo_action).item()
                    total_steps += 1

        if verbose and (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{num_epochs}: Action MSE = {total_error/total_steps:.4f}")

    model.consolidate()
    return model


def evaluate_model(
    model: nn.Module,
    dynamics: PrecisionDynamics,
    controller: PrecisionController,
    num_trajectories: int = 30,
    trajectory_length: int = 80,
    is_chunked: bool = False,
) -> Dict:
    """
    Evaluate model on precision task.

    Returns:
        Dict with action_mse, success_rate, jerk, precision_success
    """
    total_action_mse = 0.0
    total_steps = 0
    successes = 0
    precision_successes = 0
    precision_attempts = 0
    all_jerks = []

    model.eval()

    for _ in range(num_trajectories):
        model.reset()
        state = dynamics.reset()
        trajectory_success = False
        entered_precision = False
        precision_success = False
        episode_actions = []

        for step in range(trajectory_length):
            expert_action = controller(state)

            with torch.no_grad():
                if is_chunked:
                    pred_action, _, _ = model.forward(state, receding_horizon=True)
                else:
                    pred_action, _, _ = model.forward(state)

            episode_actions.append(pred_action.clone())

            action_mse = F.mse_loss(pred_action, expert_action).item()
            total_action_mse += action_mse
            total_steps += 1

            # Track precision zone performance
            if dynamics.in_precision_zone(state):
                if not entered_precision:
                    entered_precision = True
                    precision_attempts += 1

            # Execute action
            next_state = dynamics.step(state, pred_action)
            state = next_state

            if dynamics.reached_goal(state):
                trajectory_success = True
                if entered_precision:
                    precision_success = True

        if trajectory_success:
            successes += 1
        if precision_success:
            precision_successes += 1

        # Compute jerk for this trajectory
        if len(episode_actions) >= 3:
            jerk = compute_jerk(episode_actions)
            all_jerks.append(jerk)

    model.train()

    return {
        'action_mse': total_action_mse / total_steps,
        'success_rate': successes / num_trajectories,
        'jerk': np.mean(all_jerks) if all_jerks else 0.0,
        'precision_success': precision_successes / max(precision_attempts, 1),
    }


def run_chunking_test():
    """Full action chunking test."""

    print("=" * 70)
    print("ACTION CHUNKING TEST: Single-Step vs Chunked Readout")
    print("=" * 70)
    print()

    # Setup
    dynamics = PrecisionDynamics()
    controller = PrecisionController(dynamics.target)

    # Collect demos
    print("Collecting demonstrations...")
    demos = collect_demos(dynamics, controller, num_demos=50, demo_length=80)
    print(f"  Collected {len(demos)} demos, {len(demos[0])} steps each")

    # Train models
    print("\n" + "-" * 50)
    single_model = train_single_step(demos, num_epochs=50)

    print("\n" + "-" * 50)
    chunked_model = train_chunked(demos, chunk_size=5, num_epochs=50)

    # Evaluate
    print("\n" + "-" * 50)
    print("Evaluating models...")

    single_results = evaluate_model(
        single_model, dynamics, controller,
        num_trajectories=30, is_chunked=False
    )

    chunked_results = evaluate_model(
        chunked_model, dynamics, controller,
        num_trajectories=30, is_chunked=True
    )

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()
    print(f"{'Metric':<25} {'Single-Step':>15} {'Chunked (K=5)':>15} {'Winner':>12}")
    print("-" * 70)

    # Action MSE
    winner = "Chunked" if chunked_results['action_mse'] < single_results['action_mse'] else "Single"
    print(f"{'Action MSE':<25} {single_results['action_mse']:>15.4f} {chunked_results['action_mse']:>15.4f} {winner:>12}")

    # Success Rate
    winner = "Chunked" if chunked_results['success_rate'] > single_results['success_rate'] else "Single"
    print(f"{'Success Rate':<25} {single_results['success_rate']*100:>14.1f}% {chunked_results['success_rate']*100:>14.1f}% {winner:>12}")

    # Jerk (smoothness)
    winner = "Chunked" if chunked_results['jerk'] < single_results['jerk'] else "Single"
    print(f"{'Action Jerk (↓ better)':<25} {single_results['jerk']:>15.4f} {chunked_results['jerk']:>15.4f} {winner:>12}")

    # Precision Success
    winner = "Chunked" if chunked_results['precision_success'] > single_results['precision_success'] else "Single"
    print(f"{'Precision Success':<25} {single_results['precision_success']*100:>14.1f}% {chunked_results['precision_success']*100:>14.1f}% {winner:>12}")

    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    jerk_improvement = (single_results['jerk'] - chunked_results['jerk']) / single_results['jerk'] * 100
    if jerk_improvement > 0:
        print(f"✓ Chunking reduces jerk by {jerk_improvement:.1f}% (smoother control)")
    else:
        print(f"✗ Chunking increases jerk by {-jerk_improvement:.1f}%")

    precision_diff = chunked_results['precision_success'] - single_results['precision_success']
    if precision_diff > 0:
        print(f"✓ Chunking improves precision phase success by {precision_diff*100:.1f}%")
    elif precision_diff < 0:
        print(f"✗ Chunking hurts precision phase success by {-precision_diff*100:.1f}%")
    else:
        print("= Precision phase success is equal")

    success_diff = chunked_results['success_rate'] - single_results['success_rate']
    if success_diff > 0.05:
        print(f"✓ Chunking improves overall success by {success_diff*100:.1f}%")
    elif success_diff < -0.05:
        print(f"✗ Chunking hurts overall success by {-success_diff*100:.1f}%")
    else:
        print("= Overall success rates are similar")

    # Summary verdict
    print("\n" + "-" * 50)
    wins = 0
    if chunked_results['jerk'] < single_results['jerk']:
        wins += 1
    if chunked_results['precision_success'] >= single_results['precision_success']:
        wins += 1
    if chunked_results['success_rate'] >= single_results['success_rate']:
        wins += 1

    if wins >= 2:
        print("VERDICT: Chunking provides benefits for manipulation tasks")
    else:
        print("VERDICT: Chunking needs further tuning")

    return {
        'single': single_results,
        'chunked': chunked_results,
    }


if __name__ == "__main__":
    results = run_chunking_test()
