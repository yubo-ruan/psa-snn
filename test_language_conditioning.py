"""
Test Language-Conditioned PSA.

Key success criterion:
    Same observation + different instruction → different action

Test protocol:
1. Two tasks in same environment (e.g., "go left" vs "go right")
2. Train on demos for both tasks
3. Test: given same observation, does instruction change action?
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass

from psa.language_conditioned import LanguageConditionedPSA, LanguageEncoder


@dataclass
class TaskResults:
    task_name: str
    train_action_mse: float
    test_action_mse: float
    success_rate: float


class TwoTaskDynamics:
    """
    Environment with two possible tasks:
    - Task A: "go left" - move toward x=-1
    - Task B: "go right" - move toward x=+1

    State: [x, y, vx, vy]
    Action: [ax, ay]
    """

    def __init__(self, dt: float = 0.1, noise_std: float = 0.01):
        self.dt = dt
        self.noise_std = noise_std
        self.state_dim = 4
        self.action_dim = 2

    def reset(self) -> torch.Tensor:
        # Start near origin
        x = torch.rand(2) * 0.5 - 0.25  # [-0.25, 0.25]
        v = torch.zeros(2)
        return torch.cat([x, v])

    def step(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x, v = state[:2], state[2:]
        v_new = v + action * self.dt
        x_new = x + v_new * self.dt
        x_new = torch.clamp(x_new, -2, 2)
        v_new = torch.clamp(v_new, -1, 1)
        return torch.cat([x_new, v_new]) + torch.randn(4) * self.noise_std


class GoLeftController:
    """Expert for 'go left' task."""
    def __init__(self):
        self.target = torch.tensor([-1.0, 0.0])

    def __call__(self, state: torch.Tensor) -> torch.Tensor:
        x, v = state[:2], state[2:]
        error = self.target - x
        action = 2.0 * error - 0.5 * v
        return torch.clamp(action, -1, 1)


class GoRightController:
    """Expert for 'go right' task."""
    def __init__(self):
        self.target = torch.tensor([1.0, 0.0])

    def __call__(self, state: torch.Tensor) -> torch.Tensor:
        x, v = state[:2], state[2:]
        error = self.target - x
        action = 2.0 * error - 0.5 * v
        return torch.clamp(action, -1, 1)


def collect_task_demos(
    dynamics: TwoTaskDynamics,
    controller,
    task_instruction: str,
    num_demos: int = 20,
    demo_length: int = 50,
) -> List[Dict]:
    """Collect demonstrations for a task."""
    demos = []
    for _ in range(num_demos):
        trajectory = []
        state = dynamics.reset()
        for _ in range(demo_length):
            action = controller(state)
            next_state = dynamics.step(state, action)
            trajectory.append({
                'state': state.clone(),
                'action': action.clone(),
                'next_state': next_state.clone(),
                'instruction': task_instruction,
            })
            state = next_state
        demos.append(trajectory)
    return demos


def train_language_psa(
    task_demos: Dict[str, List],
    num_epochs: int = 30,
    verbose: bool = True,
) -> Tuple[LanguageConditionedPSA, Dict[str, torch.Tensor]]:
    """Train language-conditioned PSA on multiple tasks."""

    model = LanguageConditionedPSA(
        obs_dim=4,
        action_dim=2,
        num_neurons=64,
        language_dim=32,
        readout_hidden=32,
    )

    # Use orthogonal task embeddings for clear separation
    # (In real LIBERO, these would come from frozen language model)
    task_names = list(task_demos.keys())
    task_embeddings = {}
    for i, task_name in enumerate(task_names):
        embed = torch.zeros(32)
        # Make embeddings clearly different - opposite signs
        if i == 0:
            embed[:16] = 1.0
            embed[16:] = -1.0
        else:
            embed[:16] = -1.0
            embed[16:] = 1.0
        embed = embed + torch.randn(32) * 0.1  # Small noise
        task_embeddings[task_name] = embed
        if verbose:
            print(f"  Task '{task_name}' embedding norm: {embed.norm():.2f}")

    if verbose:
        print(f"Training on {len(task_demos)} tasks...")

    for epoch in range(num_epochs):
        total_action_error = 0.0
        total_state_error = 0.0
        total_steps = 0

        # Interleave tasks (important for multi-task learning)
        all_demos = []
        for task_name, demos in task_demos.items():
            for demo in demos:
                all_demos.append((task_name, demo))

        # Shuffle demos
        np.random.shuffle(all_demos)

        for task_name, demo in all_demos:
            model.reset()
            model.set_task_embedding(task_embeddings[task_name])

            for step in demo:
                state = step['state']
                demo_action = step['action']
                next_state = step['next_state']

                # Forward
                pred_action, pred_next, info = model.forward(state)

                # Local update
                model.update(next_state, demo_action)

                with torch.no_grad():
                    action_mse = F.mse_loss(pred_action, demo_action).item()
                    state_mse = F.mse_loss(pred_next, next_state).item()
                    total_action_error += action_mse
                    total_state_error += state_mse
                    total_steps += 1

        if verbose and (epoch + 1) % 5 == 0:
            avg_action = total_action_error / total_steps
            avg_state = total_state_error / total_steps
            print(f"  Epoch {epoch+1}/{num_epochs}: Action MSE={avg_action:.4f}, State MSE={avg_state:.4f}")

    model.consolidate()
    return model, task_embeddings


def test_task_switching(
    model: LanguageConditionedPSA,
    task_embeddings: Dict[str, torch.Tensor],
    dynamics: TwoTaskDynamics,
    controllers: Dict[str, callable],
    num_tests: int = 20,
) -> Dict[str, float]:
    """
    Test if model switches behavior based on instruction.

    Key metric: action divergence when given same state but different instruction.
    """
    results = {}

    # Test each task
    for task_name, controller in controllers.items():
        total_action_mse = 0.0
        successes = 0
        total_steps = 0

        model.eval()
        task_embed = task_embeddings[task_name]

        for _ in range(num_tests):
            model.reset()
            model.set_task_embedding(task_embed)

            state = dynamics.reset()
            trajectory_success = False

            for step in range(50):
                expert_action = controller(state)

                with torch.no_grad():
                    pred_action, _, _ = model.forward(state)

                action_mse = F.mse_loss(pred_action, expert_action).item()
                total_action_mse += action_mse
                total_steps += 1

                # Execute model action
                next_state = dynamics.step(state, pred_action)
                state = next_state

                # Check success (reached target)
                target_x = -1.0 if "left" in task_name else 1.0
                if abs(state[0].item() - target_x) < 0.2:
                    trajectory_success = True

            if trajectory_success:
                successes += 1

        model.train()

        results[task_name] = {
            'action_mse': total_action_mse / total_steps,
            'success_rate': successes / num_tests,
        }

    return results


def test_action_divergence(
    model: LanguageConditionedPSA,
    task_embeddings: Dict[str, torch.Tensor],
    dynamics: TwoTaskDynamics,
    num_tests: int = 100,
) -> float:
    """
    Key test: Same state, different instruction → different action?

    Returns action divergence (higher = better differentiation).
    """
    model.eval()

    task_names = list(task_embeddings.keys())
    embed_a = task_embeddings[task_names[0]]
    embed_b = task_embeddings[task_names[1]]

    divergences = []

    for _ in range(num_tests):
        # Same initial state
        state = dynamics.reset()

        # Reset and set task A
        model.reset()
        model.set_task_embedding(embed_a)
        # Warmup: run a few steps to let language effect settle
        with torch.no_grad():
            for _ in range(3):
                _, _, _ = model.forward(state)
            action_a, _, _ = model.forward(state)

        # Reset and set task B (SAME state)
        model.reset()
        model.set_task_embedding(embed_b)
        with torch.no_grad():
            for _ in range(3):
                _, _, _ = model.forward(state)
            action_b, _, _ = model.forward(state)

        # Compute divergence
        divergence = (action_a - action_b).abs().mean().item()
        divergences.append(divergence)

    model.train()
    return np.mean(divergences)


def train_mlp_baseline(
    task_demos: Dict[str, List],
    num_epochs: int = 100,
) -> nn.Module:
    """Train MLP baseline that takes (state, task_id) as input."""

    # Simple MLP with task conditioning via one-hot
    num_tasks = len(task_demos)

    mlp = nn.Sequential(
        nn.Linear(4 + num_tasks, 64),  # state + one-hot task
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 2),
        nn.Tanh(),
    )

    optimizer = torch.optim.Adam(mlp.parameters(), lr=0.01)

    # Prepare data
    all_states = []
    all_actions = []
    all_task_ids = []

    task_names = list(task_demos.keys())
    for task_idx, task_name in enumerate(task_names):
        for demo in task_demos[task_name]:
            for step in demo:
                all_states.append(step['state'])
                all_actions.append(step['action'])
                # One-hot task ID
                task_onehot = torch.zeros(num_tasks)
                task_onehot[task_idx] = 1.0
                all_task_ids.append(task_onehot)

    states = torch.stack(all_states)
    actions = torch.stack(all_actions)
    task_ids = torch.stack(all_task_ids)

    inputs = torch.cat([states, task_ids], dim=1)

    for _ in range(num_epochs):
        optimizer.zero_grad()
        pred = mlp(inputs)
        loss = F.mse_loss(pred, actions)
        loss.backward()
        optimizer.step()

    return mlp, task_names


def test_mlp_divergence(
    mlp: nn.Module,
    task_names: List[str],
    dynamics: TwoTaskDynamics,
    num_tests: int = 100,
) -> float:
    """Test MLP action divergence."""
    mlp.eval()
    num_tasks = len(task_names)
    divergences = []

    for _ in range(num_tests):
        state = dynamics.reset()

        # Task A (first task)
        task_a = torch.zeros(num_tasks)
        task_a[0] = 1.0
        input_a = torch.cat([state, task_a])

        # Task B (second task)
        task_b = torch.zeros(num_tasks)
        task_b[1] = 1.0
        input_b = torch.cat([state, task_b])

        with torch.no_grad():
            action_a = mlp(input_a)
            action_b = mlp(input_b)

        divergence = (action_a - action_b).abs().mean().item()
        divergences.append(divergence)

    mlp.train()
    return np.mean(divergences)


def run_language_test():
    """Full language conditioning test."""

    print("=" * 70)
    print("LANGUAGE-CONDITIONED PSA TEST")
    print("=" * 70)
    print()

    dynamics = TwoTaskDynamics()

    controllers = {
        "go left": GoLeftController(),
        "go right": GoRightController(),
    }

    # Collect demos
    print("Collecting demonstrations...")
    task_demos = {}
    for task_name, controller in controllers.items():
        task_demos[task_name] = collect_task_demos(
            dynamics, controller, task_name,
            num_demos=25, demo_length=50
        )
        print(f"  {task_name}: {len(task_demos[task_name])} demos")

    # Train PSA
    print("\n" + "-" * 50)
    psa_model, task_embeddings = train_language_psa(task_demos, num_epochs=50)

    # Train MLP baseline
    print("\n" + "-" * 50)
    print("Training MLP baseline...")
    mlp, task_names = train_mlp_baseline(task_demos, num_epochs=100)

    # Test task-specific performance
    print("\n" + "-" * 50)
    print("Testing task-specific performance...")

    psa_results = test_task_switching(psa_model, task_embeddings, dynamics, controllers)
    print("\nPSA Results:")
    for task, res in psa_results.items():
        print(f"  {task}: MSE={res['action_mse']:.4f}, Success={res['success_rate']*100:.1f}%")

    # Test action divergence (KEY METRIC)
    print("\n" + "-" * 50)
    print("Testing action divergence (same state, different instruction)...")

    psa_divergence = test_action_divergence(psa_model, task_embeddings, dynamics)
    mlp_divergence = test_mlp_divergence(mlp, task_names, dynamics)

    print(f"\n  PSA action divergence: {psa_divergence:.4f}")
    print(f"  MLP action divergence: {mlp_divergence:.4f}")

    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    if psa_divergence > 0.3:
        print("✓ PSA shows strong task differentiation (divergence > 0.3)")
    elif psa_divergence > 0.1:
        print("~ PSA shows moderate task differentiation (0.1 < divergence < 0.3)")
    else:
        print("✗ PSA shows weak task differentiation (divergence < 0.1)")

    # Overall success
    avg_success = np.mean([r['success_rate'] for r in psa_results.values()])
    print(f"\nOverall PSA success rate: {avg_success*100:.1f}%")

    if psa_divergence > 0.1 and avg_success > 0.5:
        print("\n✓ Language conditioning is working!")
    else:
        print("\n✗ Language conditioning needs improvement")

    return {
        'psa_results': psa_results,
        'psa_divergence': psa_divergence,
        'mlp_divergence': mlp_divergence,
    }


if __name__ == "__main__":
    results = run_language_test()
