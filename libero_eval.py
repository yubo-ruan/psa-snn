"""
LIBERO-Style Evaluation Framework for PSA.

This creates a simulated multi-task manipulation benchmark that tests
the same properties as LIBERO:
- Multi-task learning (10 tasks per suite)
- Language conditioning
- Sequential learning (for forgetting)
- Transfer across suites

Three evaluation protocols:
A. Suite-level success: Train on suite, test on all tasks
B. Forward transfer: Pretrain on Suite A, adapt to Suite B
C. Forgetting: Train A→B→C, measure retention on A

Key metrics:
- Mean success rate
- Per-task variance
- Assembly stability
- Action jerk
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field
import copy

from psa.language_conditioned import LanguageConditionedPSA
from psa.action_readout import ThreeFactorWithEligibility


# ============================================================================
# Simulated LIBERO-Style Environment
# ============================================================================

@dataclass
class TaskSpec:
    """Specification for a manipulation task."""
    name: str
    instruction: str
    target_pos: torch.Tensor
    obstacle_pos: Optional[torch.Tensor] = None
    requires_precision: bool = False
    task_embedding: Optional[torch.Tensor] = None


class ManipulationEnv:
    """
    Simulated manipulation environment.

    State: [gripper_x, gripper_y, gripper_z, gripper_open,
            object_x, object_y, object_z,
            target_x, target_y, target_z]

    Action: [dx, dy, dz, grip_action]  (4D)

    This simulates pick-and-place, pushing, and positioning tasks.
    """

    def __init__(
        self,
        task: TaskSpec,
        dt: float = 0.05,
        noise_std: float = 0.01,
        max_steps: int = 100,
    ):
        self.task = task
        self.dt = dt
        self.noise_std = noise_std
        self.max_steps = max_steps
        self.state_dim = 10
        self.action_dim = 4

        self.step_count = 0
        self.state = None

    def reset(self) -> torch.Tensor:
        """Reset environment to initial state."""
        self.step_count = 0

        # Random gripper start
        gripper = torch.rand(3) * 0.5 - 0.25
        gripper[2] = 0.3 + torch.rand(1).item() * 0.2  # Z starts higher
        gripper_open = torch.tensor([1.0])  # Open

        # Object starts in workspace
        obj = torch.rand(3) * 0.3 - 0.15
        obj[2] = 0.05  # On table

        # Target from task
        target = self.task.target_pos.clone()

        self.state = torch.cat([gripper, gripper_open, obj, target])
        return self.state.clone()

    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, float, bool, Dict]:
        """Execute action and return next state."""
        self.step_count += 1

        action = torch.clamp(action, -1, 1)

        gripper = self.state[:3]
        gripper_open = self.state[3:4]
        obj = self.state[4:7]
        target = self.state[7:10]

        # Move gripper
        gripper = gripper + action[:3] * self.dt * 2.0
        gripper = torch.clamp(gripper, -0.5, 0.5)
        gripper[2] = torch.clamp(gripper[2], 0.02, 0.5)  # Z bounds

        # Grip action
        grip_cmd = action[3]
        if grip_cmd > 0.3:
            gripper_open = torch.tensor([0.0])  # Close
        elif grip_cmd < -0.3:
            gripper_open = torch.tensor([1.0])  # Open

        # Object physics
        gripper_to_obj = (gripper - obj).norm()

        if gripper_open.item() < 0.5 and gripper_to_obj < 0.1:
            # Gripping: object follows gripper
            obj = gripper.clone()
            obj[2] = max(obj[2] - 0.05, 0.05)  # Slightly below gripper

        # Gravity when not gripped
        if gripper_open.item() > 0.5 or gripper_to_obj > 0.1:
            if obj[2] > 0.05:
                obj[2] = max(obj[2] - 0.02, 0.05)

        # Add noise
        obj = obj + torch.randn(3) * self.noise_std

        self.state = torch.cat([gripper, gripper_open, obj, target])

        # Compute reward/success
        obj_to_target = (obj - target).norm()
        success = obj_to_target < 0.08

        reward = -obj_to_target.item()
        if success:
            reward += 1.0

        done = success or self.step_count >= self.max_steps

        info = {
            'success': success,
            'obj_to_target': obj_to_target.item(),
            'gripper_to_obj': gripper_to_obj.item(),
        }

        return self.state.clone(), reward, done, info


class ExpertPolicy:
    """Simple expert for pick-and-place tasks."""

    def __call__(self, state: torch.Tensor) -> torch.Tensor:
        gripper = state[:3]
        gripper_open = state[3]
        obj = state[4:7]
        target = state[7:10]

        gripper_to_obj = (gripper[:2] - obj[:2]).norm()
        gripper_height = gripper[2]
        obj_to_target = (obj - target).norm()

        # Phase 1: Move above object
        if gripper_to_obj > 0.08 or gripper_height > obj[2] + 0.15:
            # Move toward object XY
            action_xy = (obj[:2] - gripper[:2]) * 3.0
            action_z = torch.tensor([-0.5]) if gripper_height > obj[2] + 0.1 else torch.tensor([0.0])
            grip = torch.tensor([-1.0])  # Keep open
            return torch.cat([action_xy, action_z, grip])

        # Phase 2: Grasp
        if gripper_open > 0.5:
            return torch.tensor([0.0, 0.0, -0.5, 1.0])  # Lower and close

        # Phase 3: Lift and move to target
        if gripper_height < 0.25:
            return torch.tensor([0.0, 0.0, 1.0, 0.5])  # Lift

        # Phase 4: Move to target XY
        if (gripper[:2] - target[:2]).norm() > 0.05:
            action_xy = (target[:2] - gripper[:2]) * 3.0
            return torch.cat([action_xy, torch.tensor([0.0, 0.5])])

        # Phase 5: Lower and release
        if gripper_height > target[2] + 0.08:
            return torch.tensor([0.0, 0.0, -0.5, 0.5])

        # Release
        return torch.tensor([0.0, 0.0, 0.0, -1.0])


# ============================================================================
# Suite Definition
# ============================================================================

def create_suite(name: str, num_tasks: int = 10) -> List[TaskSpec]:
    """Create a suite of related tasks."""
    tasks = []

    # Create diverse target positions based on suite
    if name == "spatial":
        # Spatial suite: different target locations
        angles = torch.linspace(0, 2 * np.pi, num_tasks + 1)[:-1]
        for i, angle in enumerate(angles):
            x = 0.3 * torch.cos(angle)
            y = 0.3 * torch.sin(angle)
            z = 0.05 + 0.1 * (i % 3)

            tasks.append(TaskSpec(
                name=f"place_at_pos_{i}",
                instruction=f"place the object at position {i}",
                target_pos=torch.tensor([x.item(), y.item(), z]),
                requires_precision=(i % 2 == 0),
            ))

    elif name == "object":
        # Object suite: different object types (simulated via target variance)
        for i in range(num_tasks):
            x = 0.2 * (i % 5 - 2) / 2
            y = 0.2 * (i // 5 - 0.5)
            z = 0.05

            tasks.append(TaskSpec(
                name=f"pick_object_{i}",
                instruction=f"pick up object {i} and place it in the bin",
                target_pos=torch.tensor([x, y, z]),
            ))

    elif name == "goal":
        # Goal suite: different goal containers (simulated)
        goals = ["red bin", "blue bin", "green box", "yellow tray", "white plate",
                 "metal bowl", "wooden box", "glass jar", "plastic container", "paper bag"]
        for i, goal in enumerate(goals[:num_tasks]):
            angle = i * 2 * np.pi / num_tasks
            x = 0.25 * np.cos(angle)
            y = 0.25 * np.sin(angle)

            tasks.append(TaskSpec(
                name=f"place_in_{goal.replace(' ', '_')}",
                instruction=f"place the object in the {goal}",
                target_pos=torch.tensor([x, y, 0.05]),
            ))

    # Create task embeddings (simulating frozen language model)
    for i, task in enumerate(tasks):
        embed = torch.zeros(32)
        # Create structured embeddings based on task index
        embed[i % 16] = 1.0
        embed[16 + (i * 7) % 16] = 1.0
        embed = embed + torch.randn(32) * 0.1
        task.task_embedding = embed

    return tasks


# ============================================================================
# Evaluation Functions
# ============================================================================

def collect_demos_for_task(
    task: TaskSpec,
    num_demos: int = 10,
    max_steps: int = 100,
) -> List[List[Tuple]]:
    """Collect expert demonstrations for a task."""
    env = ManipulationEnv(task, max_steps=max_steps)
    expert = ExpertPolicy()
    demos = []

    for _ in range(num_demos):
        trajectory = []
        state = env.reset()

        for _ in range(max_steps):
            action = expert(state)
            next_state, _, done, _ = env.step(action)
            trajectory.append((state.clone(), action.clone(), next_state.clone()))
            state = next_state
            if done:
                break

        demos.append(trajectory)

    return demos


def train_on_suite(
    model: LanguageConditionedPSA,
    suite: List[TaskSpec],
    demos_per_task: int = 10,
    num_epochs: int = 30,
    verbose: bool = True,
) -> Dict:
    """Train model on all tasks in a suite."""

    # Collect demos for all tasks
    all_demos = {}
    for task in suite:
        all_demos[task.name] = {
            'demos': collect_demos_for_task(task, demos_per_task),
            'task': task,
        }

    if verbose:
        print(f"Training on {len(suite)} tasks, {demos_per_task} demos each...")

    train_history = []

    for epoch in range(num_epochs):
        total_action_error = 0.0
        total_steps = 0

        # Shuffle task order each epoch
        task_order = list(all_demos.keys())
        np.random.shuffle(task_order)

        for task_name in task_order:
            task_data = all_demos[task_name]
            task = task_data['task']
            demos = task_data['demos']

            for demo in demos:
                model.reset()
                model.set_task_embedding(task.task_embedding)

                for state, demo_action, next_state in demo:
                    pred_action, pred_next, _ = model.forward(state)
                    model.update(next_state, demo_action)

                    with torch.no_grad():
                        total_action_error += F.mse_loss(pred_action, demo_action).item()
                        total_steps += 1

        avg_error = total_action_error / total_steps
        train_history.append(avg_error)

        if verbose and (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{num_epochs}: Action MSE = {avg_error:.4f}")

    model.consolidate()

    return {
        'train_history': train_history,
        'final_mse': train_history[-1],
    }


def evaluate_on_suite(
    model: LanguageConditionedPSA,
    suite: List[TaskSpec],
    num_eval_episodes: int = 10,
    max_steps: int = 100,
    verbose: bool = True,
) -> Dict:
    """Evaluate model on all tasks in a suite."""

    model.eval()
    results = {}
    all_jerks = []

    for task in suite:
        env = ManipulationEnv(task, max_steps=max_steps)
        successes = 0
        task_jerks = []

        for _ in range(num_eval_episodes):
            model.reset()
            model.set_task_embedding(task.task_embedding)

            state = env.reset()
            episode_actions = []

            for _ in range(max_steps):
                with torch.no_grad():
                    action, _, _ = model.forward(state)

                episode_actions.append(action.clone())
                state, _, done, info = env.step(action)

                if done:
                    if info['success']:
                        successes += 1
                    break

            # Compute jerk
            if len(episode_actions) >= 3:
                jerk = sum(
                    (episode_actions[i] - 2*episode_actions[i-1] + episode_actions[i-2]).norm().item()
                    for i in range(2, len(episode_actions))
                ) / (len(episode_actions) - 2)
                task_jerks.append(jerk)

        results[task.name] = {
            'success_rate': successes / num_eval_episodes,
            'mean_jerk': np.mean(task_jerks) if task_jerks else 0.0,
        }
        all_jerks.extend(task_jerks)

    model.train()

    # Aggregate stats
    success_rates = [r['success_rate'] for r in results.values()]

    return {
        'per_task': results,
        'mean_success': np.mean(success_rates),
        'std_success': np.std(success_rates),
        'min_success': np.min(success_rates),
        'max_success': np.max(success_rates),
        'mean_jerk': np.mean(all_jerks) if all_jerks else 0.0,
    }


# ============================================================================
# Main Evaluation Protocols
# ============================================================================

def eval_suite_level(verbose: bool = True) -> Dict:
    """
    Protocol A: Suite-level success evaluation.

    Train on one full suite, evaluate on all tasks.
    """
    if verbose:
        print("=" * 70)
        print("PROTOCOL A: Suite-Level Success")
        print("=" * 70)
        print()

    # Create model
    model = LanguageConditionedPSA(
        obs_dim=10,
        action_dim=4,
        num_neurons=128,
        language_dim=32,
        readout_hidden=64,
    )

    # Create suite
    suite = create_suite("spatial", num_tasks=10)
    if verbose:
        print(f"Created suite with {len(suite)} tasks")

    # Train
    train_results = train_on_suite(model, suite, demos_per_task=10, num_epochs=30, verbose=verbose)

    # Evaluate
    if verbose:
        print("\nEvaluating on suite...")

    eval_results = evaluate_on_suite(model, suite, num_eval_episodes=10, verbose=verbose)

    if verbose:
        print("\n" + "-" * 50)
        print("Results:")
        print(f"  Mean Success: {eval_results['mean_success']*100:.1f}%")
        print(f"  Std Success:  {eval_results['std_success']*100:.1f}%")
        print(f"  Min Success:  {eval_results['min_success']*100:.1f}%")
        print(f"  Max Success:  {eval_results['max_success']*100:.1f}%")
        print(f"  Mean Jerk:    {eval_results['mean_jerk']:.4f}")

    return {
        'train': train_results,
        'eval': eval_results,
    }


def eval_forward_transfer(verbose: bool = True) -> Dict:
    """
    Protocol B: Forward transfer evaluation.

    Pretrain on Suite A, then adapt to Suite B (readout only).
    Compare vs training from scratch.
    """
    if verbose:
        print("=" * 70)
        print("PROTOCOL B: Forward Transfer")
        print("=" * 70)
        print()

    # Create suites
    suite_a = create_suite("spatial", num_tasks=10)
    suite_b = create_suite("object", num_tasks=10)

    if verbose:
        print(f"Suite A (pretrain): {len(suite_a)} tasks")
        print(f"Suite B (transfer): {len(suite_b)} tasks")

    # === Pretrained model ===
    if verbose:
        print("\n--- Pretrained + Transfer ---")

    pretrained_model = LanguageConditionedPSA(
        obs_dim=10,
        action_dim=4,
        num_neurons=128,
        language_dim=32,
        readout_hidden=64,
    )

    # Pretrain on Suite A
    if verbose:
        print("Pretraining on Suite A...")
    train_on_suite(pretrained_model, suite_a, demos_per_task=10, num_epochs=30, verbose=False)

    # Freeze PSA, only train readout
    for param in pretrained_model.psa.parameters():
        param.requires_grad = False

    # Adapt to Suite B (fewer epochs)
    if verbose:
        print("Adapting to Suite B (readout only)...")
    train_on_suite(pretrained_model, suite_b, demos_per_task=10, num_epochs=15, verbose=False)

    pretrained_results = evaluate_on_suite(pretrained_model, suite_b, verbose=False)

    # === From scratch model ===
    if verbose:
        print("\n--- From Scratch ---")

    scratch_model = LanguageConditionedPSA(
        obs_dim=10,
        action_dim=4,
        num_neurons=128,
        language_dim=32,
        readout_hidden=64,
    )

    # Train on Suite B only
    if verbose:
        print("Training on Suite B from scratch...")
    train_on_suite(scratch_model, suite_b, demos_per_task=10, num_epochs=30, verbose=False)

    scratch_results = evaluate_on_suite(scratch_model, suite_b, verbose=False)

    if verbose:
        print("\n" + "-" * 50)
        print("Results on Suite B:")
        print(f"  Pretrained + Transfer: {pretrained_results['mean_success']*100:.1f}%")
        print(f"  From Scratch:          {scratch_results['mean_success']*100:.1f}%")

        if pretrained_results['mean_success'] > scratch_results['mean_success']:
            improvement = (pretrained_results['mean_success'] - scratch_results['mean_success']) * 100
            print(f"  ✓ Transfer improves by {improvement:.1f}%")
        else:
            print(f"  ✗ Transfer did not help")

    return {
        'pretrained': pretrained_results,
        'scratch': scratch_results,
    }


def eval_forgetting(verbose: bool = True) -> Dict:
    """
    Protocol C: Catastrophic forgetting evaluation.

    Train sequentially: Suite A → B → C
    Measure retention on A after finishing C.
    """
    if verbose:
        print("=" * 70)
        print("PROTOCOL C: Catastrophic Forgetting")
        print("=" * 70)
        print()

    # Create three suites
    suite_a = create_suite("spatial", num_tasks=10)
    suite_b = create_suite("object", num_tasks=10)
    suite_c = create_suite("goal", num_tasks=10)

    if verbose:
        print(f"Suite A: {len(suite_a)} tasks")
        print(f"Suite B: {len(suite_b)} tasks")
        print(f"Suite C: {len(suite_c)} tasks")

    # === PSA Model ===
    if verbose:
        print("\n--- PSA (Local Learning) ---")

    psa_model = LanguageConditionedPSA(
        obs_dim=10,
        action_dim=4,
        num_neurons=128,
        language_dim=32,
        readout_hidden=64,
    )

    # Train A, eval A
    if verbose:
        print("Training on Suite A...")
    train_on_suite(psa_model, suite_a, demos_per_task=10, num_epochs=20, verbose=False)
    psa_after_a = evaluate_on_suite(psa_model, suite_a, verbose=False)['mean_success']
    if verbose:
        print(f"  After A: {psa_after_a*100:.1f}% on A")

    # Train B, eval A
    if verbose:
        print("Training on Suite B...")
    train_on_suite(psa_model, suite_b, demos_per_task=10, num_epochs=20, verbose=False)
    psa_after_b = evaluate_on_suite(psa_model, suite_a, verbose=False)['mean_success']
    if verbose:
        print(f"  After B: {psa_after_b*100:.1f}% on A")

    # Train C, eval A
    if verbose:
        print("Training on Suite C...")
    train_on_suite(psa_model, suite_c, demos_per_task=10, num_epochs=20, verbose=False)
    psa_after_c = evaluate_on_suite(psa_model, suite_a, verbose=False)['mean_success']
    if verbose:
        print(f"  After C: {psa_after_c*100:.1f}% on A")

    psa_forgetting = psa_after_a - psa_after_c

    # === MLP Baseline ===
    if verbose:
        print("\n--- MLP Baseline ---")

    # Simple MLP for comparison
    mlp_results = simulate_mlp_forgetting(suite_a, suite_b, suite_c, verbose)

    if verbose:
        print("\n" + "-" * 50)
        print("Forgetting Results (Suite A performance):")
        print(f"  PSA: {psa_after_a*100:.1f}% → {psa_after_c*100:.1f}% (forgot {psa_forgetting*100:.1f}%)")
        print(f"  MLP: {mlp_results['after_a']*100:.1f}% → {mlp_results['after_c']*100:.1f}% (forgot {mlp_results['forgetting']*100:.1f}%)")

        if psa_forgetting < mlp_results['forgetting']:
            print(f"  ✓ PSA forgets {(mlp_results['forgetting'] - psa_forgetting)*100:.1f}% less")
        else:
            print(f"  ✗ PSA forgets more than MLP")

    return {
        'psa': {
            'after_a': psa_after_a,
            'after_b': psa_after_b,
            'after_c': psa_after_c,
            'forgetting': psa_forgetting,
        },
        'mlp': mlp_results,
    }


def simulate_mlp_forgetting(suite_a, suite_b, suite_c, verbose=True) -> Dict:
    """
    Simulate MLP baseline forgetting.

    Uses a simple MLP with gradient descent (more prone to catastrophic forgetting).
    """

    class SimpleMLP(nn.Module):
        def __init__(self, obs_dim, action_dim, language_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(obs_dim + language_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, action_dim),
                nn.Tanh(),
            )

        def forward(self, obs, language):
            x = torch.cat([obs, language], dim=-1)
            return self.net(x)

    model = SimpleMLP(10, 4, 32)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    def train_mlp_on_suite(suite, epochs=20):
        for _ in range(epochs):
            for task in suite:
                demos = collect_demos_for_task(task, num_demos=5)
                for demo in demos:
                    for state, action, _ in demo:
                        pred = model(state, task.task_embedding)
                        loss = F.mse_loss(pred, action)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

    def eval_mlp_on_suite(suite):
        model.eval()
        successes = 0
        total = 0

        for task in suite:
            env = ManipulationEnv(task)
            for _ in range(5):
                state = env.reset()
                for _ in range(100):
                    with torch.no_grad():
                        action = model(state, task.task_embedding)
                    state, _, done, info = env.step(action)
                    if done:
                        if info['success']:
                            successes += 1
                        break
                total += 1

        model.train()
        return successes / total

    # Sequential training
    if verbose:
        print("Training on Suite A...")
    train_mlp_on_suite(suite_a)
    after_a = eval_mlp_on_suite(suite_a)
    if verbose:
        print(f"  After A: {after_a*100:.1f}% on A")

    if verbose:
        print("Training on Suite B...")
    train_mlp_on_suite(suite_b)
    after_b = eval_mlp_on_suite(suite_a)
    if verbose:
        print(f"  After B: {after_b*100:.1f}% on A")

    if verbose:
        print("Training on Suite C...")
    train_mlp_on_suite(suite_c)
    after_c = eval_mlp_on_suite(suite_a)
    if verbose:
        print(f"  After C: {after_c*100:.1f}% on A")

    return {
        'after_a': after_a,
        'after_b': after_b,
        'after_c': after_c,
        'forgetting': after_a - after_c,
    }


# ============================================================================
# Main
# ============================================================================

def run_all_evaluations():
    """Run all three evaluation protocols."""

    print("=" * 70)
    print("LIBERO-STYLE PSA EVALUATION")
    print("=" * 70)
    print()

    results = {}

    # Protocol A: Suite-level
    results['suite_level'] = eval_suite_level(verbose=True)
    print()

    # Protocol B: Forward transfer
    results['forward_transfer'] = eval_forward_transfer(verbose=True)
    print()

    # Protocol C: Forgetting
    results['forgetting'] = eval_forgetting(verbose=True)
    print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print(f"Suite-Level Mean Success:   {results['suite_level']['eval']['mean_success']*100:.1f}%")
    print(f"Forward Transfer Benefit:   {(results['forward_transfer']['pretrained']['mean_success'] - results['forward_transfer']['scratch']['mean_success'])*100:.1f}%")
    print(f"PSA Forgetting:             {results['forgetting']['psa']['forgetting']*100:.1f}%")
    print(f"MLP Forgetting:             {results['forgetting']['mlp']['forgetting']*100:.1f}%")

    return results


if __name__ == "__main__":
    results = run_all_evaluations()
