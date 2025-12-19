"""
Test Eligibility Traces for Temporal Credit Assignment.

Key success criterion:
    Eligibility traces should help when there's delay between
    action and its effect being observed.

Test protocol:
1. Task with delayed feedback (effect visible N steps after action)
2. Compare: no traces, traces with different decay rates
3. Measure: learning speed, final performance, stability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass

from psa.action_readout import (
    PSAWithActionReadout,
    PSAWithEligibilityReadout,
    ThreeFactorReadout,
    ThreeFactorWithEligibility,
)


@dataclass
class EligibilityResults:
    name: str
    final_action_mse: float
    learning_curve: List[float]
    success_rate: float
    eligibility_decay: float


class DelayedEffectDynamics:
    """
    Environment where actions have delayed effects.

    This simulates manipulation tasks where:
    - Grasp actions don't instantly result in grasp
    - Push actions take time to move objects
    - The effect of an action is observed N steps later

    State: [x, y, vx, vy, delayed_force_x, delayed_force_y]
    Action: [force_x, force_y]

    The key: action at t affects state at t+delay
    """

    def __init__(
        self,
        dt: float = 0.1,
        noise_std: float = 0.01,
        delay_steps: int = 3,  # Action effect delayed by N steps
    ):
        self.dt = dt
        self.noise_std = noise_std
        self.delay_steps = delay_steps
        self.state_dim = 4
        self.action_dim = 2

        # Action buffer for delay
        self.action_buffer = []

    def reset(self) -> torch.Tensor:
        # Random start position
        x = torch.rand(2) * 1.5 - 0.75  # [-0.75, 0.75]
        v = torch.zeros(2)
        # Clear action buffer
        self.action_buffer = [torch.zeros(2) for _ in range(self.delay_steps)]
        return torch.cat([x, v])

    def step(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x, v = state[:2], state[2:]

        # Get delayed action from buffer
        delayed_action = self.action_buffer[0]
        self.action_buffer = self.action_buffer[1:] + [action.clone()]

        # Physics with delayed action
        v_new = 0.9 * v + 0.1 * delayed_action  # Velocity smoothing
        x_new = x + v_new * self.dt
        x_new = torch.clamp(x_new, -2, 2)
        v_new = torch.clamp(v_new, -1, 1)

        return torch.cat([x_new, v_new]) + torch.randn(4) * self.noise_std


class DelayedController:
    """
    Expert controller that accounts for delay.

    Since we know the delay, the expert predicts where the object
    will be and plans accordingly.
    """

    def __init__(self, target: torch.Tensor, delay_steps: int = 3):
        self.target = target
        self.delay_steps = delay_steps

    def __call__(self, state: torch.Tensor) -> torch.Tensor:
        x, v = state[:2], state[2:]

        # Predict where we'll be after delay
        future_x = x + v * self.delay_steps * 0.1

        # Control toward target from predicted position
        error = self.target - future_x
        action = 2.0 * error - 0.5 * v
        return torch.clamp(action, -1, 1)


def collect_delayed_demos(
    dynamics: DelayedEffectDynamics,
    controller: DelayedController,
    num_demos: int = 50,
    demo_length: int = 60,
) -> List[List[Tuple]]:
    """Collect demos accounting for delay."""
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


def train_model(
    model: nn.Module,
    trajectories: List,
    num_epochs: int = 50,
    is_eligibility: bool = False,
) -> List[float]:
    """Train model and return learning curve."""
    learning_curve = []

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

        avg_error = total_error / total_steps
        learning_curve.append(avg_error)

    model.consolidate()
    return learning_curve


def evaluate_model(
    model: nn.Module,
    dynamics: DelayedEffectDynamics,
    controller: DelayedController,
    num_trajectories: int = 30,
    trajectory_length: int = 60,
) -> Tuple[float, float]:
    """Evaluate model on delayed task."""
    total_action_mse = 0.0
    total_steps = 0
    successes = 0

    model.eval()

    for _ in range(num_trajectories):
        model.reset()
        state = dynamics.reset()
        trajectory_success = False

        for step in range(trajectory_length):
            expert_action = controller(state)

            with torch.no_grad():
                pred_action, _, _ = model.forward(state)

            action_mse = F.mse_loss(pred_action, expert_action).item()
            total_action_mse += action_mse
            total_steps += 1

            next_state = dynamics.step(state, pred_action)
            state = next_state

            # Success: reached target
            if (state[:2] - controller.target).norm() < 0.15:
                trajectory_success = True

        if trajectory_success:
            successes += 1

    model.train()

    return total_action_mse / total_steps, successes / num_trajectories


def run_eligibility_test():
    """Full eligibility trace test."""

    print("=" * 70)
    print("ELIGIBILITY TRACE TEST: Temporal Credit Assignment")
    print("=" * 70)
    print()

    # Setup with delayed dynamics
    delay_steps = 3
    dynamics = DelayedEffectDynamics(delay_steps=delay_steps)
    target = torch.tensor([0.8, 0.0])
    controller = DelayedController(target, delay_steps=delay_steps)

    print(f"Testing with action delay = {delay_steps} steps")
    print()

    # Collect demos
    print("Collecting demonstrations...")
    demos = collect_delayed_demos(dynamics, controller, num_demos=50, demo_length=60)

    # Test different configurations
    configs = [
        ("No Eligibility (baseline)", 0.0, False),
        ("Eligibility decay=0.5", 0.5, True),
        ("Eligibility decay=0.7", 0.7, True),
        ("Eligibility decay=0.9", 0.9, True),
    ]

    results = []

    for name, decay, use_eligibility in configs:
        print(f"\n{'-'*50}")
        print(f"Training: {name}")

        if use_eligibility:
            model = PSAWithEligibilityReadout(
                obs_dim=4,
                action_dim=2,
                psa_neurons=64,
                readout_hidden=32,
                eligibility_decay=decay,
            )
        else:
            model = PSAWithActionReadout(
                obs_dim=4,
                action_dim=2,
                psa_neurons=64,
                readout_hidden=32,
            )

        learning_curve = train_model(model, demos, num_epochs=50, is_eligibility=use_eligibility)

        print(f"  Epoch 10: MSE={learning_curve[9]:.4f}")
        print(f"  Epoch 50: MSE={learning_curve[49]:.4f}")

        final_mse, success_rate = evaluate_model(model, dynamics, controller)
        print(f"  Test MSE: {final_mse:.4f}, Success: {success_rate*100:.1f}%")

        results.append(EligibilityResults(
            name=name,
            final_action_mse=final_mse,
            learning_curve=learning_curve,
            success_rate=success_rate,
            eligibility_decay=decay,
        ))

    # Print comparison
    print("\n" + "=" * 70)
    print("RESULTS COMPARISON")
    print("=" * 70)
    print()
    print(f"{'Method':<30} {'Final MSE':>12} {'Success %':>12} {'Learn Speed':>12}")
    print("-" * 70)

    baseline_mse = results[0].final_action_mse

    for r in results:
        # Learning speed: MSE at epoch 10 / MSE at epoch 50 (higher = faster convergence)
        learn_speed = r.learning_curve[9] / (r.learning_curve[49] + 1e-6)
        improvement = (baseline_mse - r.final_action_mse) / baseline_mse * 100 if r.eligibility_decay > 0 else 0

        print(f"{r.name:<30} {r.final_action_mse:>12.4f} {r.success_rate*100:>11.1f}% {learn_speed:>12.2f}")

    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    baseline = results[0]
    best_eligibility = min(results[1:], key=lambda r: r.final_action_mse)

    mse_improvement = (baseline.final_action_mse - best_eligibility.final_action_mse) / baseline.final_action_mse * 100
    success_improvement = best_eligibility.success_rate - baseline.success_rate

    if mse_improvement > 5:
        print(f"✓ Eligibility traces reduce MSE by {mse_improvement:.1f}%")
    elif mse_improvement > 0:
        print(f"~ Eligibility traces reduce MSE by {mse_improvement:.1f}% (modest)")
    else:
        print(f"✗ Eligibility traces increase MSE by {-mse_improvement:.1f}%")

    if success_improvement > 0.05:
        print(f"✓ Eligibility traces improve success rate by {success_improvement*100:.1f}%")
    elif success_improvement > 0:
        print(f"~ Eligibility traces improve success rate by {success_improvement*100:.1f}%")
    else:
        print(f"✗ Eligibility traces hurt success rate by {-success_improvement*100:.1f}%")

    print(f"\nBest eligibility decay: {best_eligibility.eligibility_decay}")
    print(f"Best success rate: {best_eligibility.success_rate*100:.1f}%")

    # Verdict
    print("\n" + "-" * 50)
    if best_eligibility.success_rate > baseline.success_rate or best_eligibility.final_action_mse < baseline.final_action_mse:
        print("VERDICT: Eligibility traces help with delayed effects")
    else:
        print("VERDICT: Eligibility traces need tuning for this task")

    return results


if __name__ == "__main__":
    results = run_eligibility_test()
