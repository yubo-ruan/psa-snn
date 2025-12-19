"""
LIBERO Evaluation Script for PSA-SNN.

Run this on GCP with LIBERO installed.

Evaluates:
A. Suite-level success (10 tasks)
B. Forward transfer (pretrain Suite A → adapt Suite B)
C. Forgetting (sequential A → B → C)

Usage:
    python run_libero_eval.py --suite libero_spatial
    python run_libero_eval.py --eval all
    python run_libero_eval.py --eval forgetting --suites spatial object goal
"""

import argparse
import json
import os
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np

import torch
import torch.nn.functional as F

# Try to import LIBERO - fall back to simulation if not available
try:
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv
    LIBERO_AVAILABLE = True
except ImportError:
    print("LIBERO not installed. Using simulated environment.")
    LIBERO_AVAILABLE = False

from psa.language_conditioned import LanguageConditionedPSA


# ============================================================================
# LIBERO Interface
# ============================================================================

class LIBEROAdapter:
    """Adapter for LIBERO benchmark."""

    def __init__(self, suite_name: str = "libero_spatial"):
        self.suite_name = suite_name

        if LIBERO_AVAILABLE:
            self.benchmark = benchmark.get_benchmark(suite_name)
            self.task_names = self.benchmark.get_task_names()
            self.num_tasks = len(self.task_names)
        else:
            # Simulation fallback
            self.task_names = [f"task_{i}" for i in range(10)]
            self.num_tasks = 10

        print(f"Loaded suite '{suite_name}' with {self.num_tasks} tasks")

    def get_task_embedding(self, task_idx: int) -> torch.Tensor:
        """Get language embedding for task."""
        # Create structured embedding based on task
        embed = torch.zeros(32)
        embed[task_idx % 16] = 1.0
        embed[16 + (task_idx * 7) % 16] = 1.0
        embed = embed + torch.randn(32) * 0.1
        return embed

    def collect_demos(
        self,
        task_idx: int,
        num_demos: int = 10,
    ) -> List[List[tuple]]:
        """Collect demonstrations for a task."""
        if LIBERO_AVAILABLE:
            return self._collect_libero_demos(task_idx, num_demos)
        else:
            return self._collect_simulated_demos(task_idx, num_demos)

    def _collect_libero_demos(self, task_idx: int, num_demos: int):
        """Collect demos from actual LIBERO."""
        task = self.benchmark.get_task(task_idx)
        task_name = self.task_names[task_idx]

        # Load pre-collected demos from LIBERO dataset
        demos = []
        demo_files = self.benchmark.get_task_demonstration(task_idx)

        for demo_file in demo_files[:num_demos]:
            import h5py
            with h5py.File(demo_file, 'r') as f:
                obs = f['obs'][:]
                actions = f['actions'][:]

            trajectory = []
            for i in range(len(actions)):
                state = torch.tensor(obs[i], dtype=torch.float32)
                action = torch.tensor(actions[i], dtype=torch.float32)
                next_state = torch.tensor(obs[i+1] if i+1 < len(obs) else obs[i], dtype=torch.float32)
                trajectory.append((state, action, next_state))

            demos.append(trajectory)

        return demos

    def _collect_simulated_demos(self, task_idx: int, num_demos: int):
        """Generate simulated demos for testing."""
        demos = []
        for _ in range(num_demos):
            trajectory = []
            state = torch.randn(10)  # Simulated state

            for _ in range(50):  # 50 steps per demo
                # Simple expert policy
                action = -0.1 * state[:4] + torch.randn(4) * 0.1
                action = torch.clamp(action, -1, 1)
                next_state = state + action[:4].repeat(2)[:state.shape[0]] * 0.1 + torch.randn(10) * 0.01
                trajectory.append((state.clone(), action.clone(), next_state.clone()))
                state = next_state

            demos.append(trajectory)

        return demos

    def evaluate(
        self,
        model: LanguageConditionedPSA,
        task_idx: int,
        num_episodes: int = 10,
    ) -> Dict:
        """Evaluate model on a task."""
        if LIBERO_AVAILABLE:
            return self._evaluate_libero(model, task_idx, num_episodes)
        else:
            return self._evaluate_simulated(model, task_idx, num_episodes)

    def _evaluate_libero(self, model, task_idx, num_episodes):
        """Evaluate on actual LIBERO environment."""
        task = self.benchmark.get_task(task_idx)
        task_name = self.task_names[task_idx]

        # Create environment
        env = OffScreenRenderEnv(
            bddl_file=task.bddl_file,
            camera_heights=128,
            camera_widths=128,
        )

        successes = 0
        total_reward = 0
        jerks = []

        task_embed = self.get_task_embedding(task_idx)

        for ep in range(num_episodes):
            model.reset()
            model.set_task_embedding(task_embed)

            obs = env.reset()
            obs_tensor = torch.tensor(obs['robot0_proprio-state'], dtype=torch.float32)

            episode_actions = []
            done = False
            step = 0

            while not done and step < 200:
                with torch.no_grad():
                    action, _, _ = model.forward(obs_tensor)

                episode_actions.append(action.clone())

                obs, reward, done, info = env.step(action.numpy())
                obs_tensor = torch.tensor(obs['robot0_proprio-state'], dtype=torch.float32)

                total_reward += reward
                step += 1

                if info.get('success', False):
                    successes += 1
                    break

            # Compute jerk
            if len(episode_actions) >= 3:
                jerk = sum(
                    (episode_actions[i] - 2*episode_actions[i-1] + episode_actions[i-2]).norm().item()
                    for i in range(2, len(episode_actions))
                ) / (len(episode_actions) - 2)
                jerks.append(jerk)

        env.close()

        return {
            'success_rate': successes / num_episodes,
            'mean_reward': total_reward / num_episodes,
            'mean_jerk': np.mean(jerks) if jerks else 0.0,
            'task_name': task_name,
        }

    def _evaluate_simulated(self, model, task_idx, num_episodes):
        """Simulated evaluation for testing."""
        successes = 0
        jerks = []
        task_embed = self.get_task_embedding(task_idx)

        for _ in range(num_episodes):
            model.reset()
            model.set_task_embedding(task_embed)

            state = torch.randn(10)
            episode_actions = []

            for step in range(50):
                with torch.no_grad():
                    action, _, _ = model.forward(state)

                episode_actions.append(action.clone())
                state = state + action[:4].repeat(2)[:state.shape[0]] * 0.1

                # Simple success criterion
                if state.norm() < 0.5:
                    successes += 1
                    break

            if len(episode_actions) >= 3:
                jerk = sum(
                    (episode_actions[i] - 2*episode_actions[i-1] + episode_actions[i-2]).norm().item()
                    for i in range(2, len(episode_actions))
                ) / (len(episode_actions) - 2)
                jerks.append(jerk)

        return {
            'success_rate': successes / num_episodes,
            'mean_jerk': np.mean(jerks) if jerks else 0.0,
            'task_name': self.task_names[task_idx],
        }


# ============================================================================
# Evaluation Protocols
# ============================================================================

def train_on_suite(
    model: LanguageConditionedPSA,
    adapter: LIBEROAdapter,
    demos_per_task: int = 10,
    num_epochs: int = 30,
    verbose: bool = True,
):
    """Train model on all tasks in a suite."""
    # Collect demos
    all_demos = {}
    for task_idx in range(adapter.num_tasks):
        demos = adapter.collect_demos(task_idx, demos_per_task)
        embed = adapter.get_task_embedding(task_idx)
        all_demos[task_idx] = {'demos': demos, 'embedding': embed}

        if verbose:
            print(f"  Collected {len(demos)} demos for {adapter.task_names[task_idx]}")

    # Train
    for epoch in range(num_epochs):
        total_error = 0
        total_steps = 0

        task_order = list(range(adapter.num_tasks))
        np.random.shuffle(task_order)

        for task_idx in task_order:
            task_data = all_demos[task_idx]

            for demo in task_data['demos']:
                model.reset()
                model.set_task_embedding(task_data['embedding'])

                for state, action, next_state in demo:
                    pred_action, _, _ = model.forward(state)
                    model.update(next_state, action)

                    with torch.no_grad():
                        total_error += F.mse_loss(pred_action, action).item()
                        total_steps += 1

        if verbose and (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{num_epochs}: MSE = {total_error/total_steps:.4f}")

    model.consolidate()


def eval_suite_level(suite_name: str = "libero_spatial", verbose: bool = True) -> Dict:
    """Protocol A: Suite-level success evaluation."""
    if verbose:
        print("=" * 70)
        print(f"PROTOCOL A: Suite-Level Success ({suite_name})")
        print("=" * 70)

    adapter = LIBEROAdapter(suite_name)

    model = LanguageConditionedPSA(
        obs_dim=10,
        action_dim=4,
        num_neurons=128,
        language_dim=32,
        readout_hidden=64,
    )

    if verbose:
        print("\nTraining...")
    train_on_suite(model, adapter, demos_per_task=10, num_epochs=30, verbose=verbose)

    if verbose:
        print("\nEvaluating...")

    results = {}
    for task_idx in range(adapter.num_tasks):
        task_result = adapter.evaluate(model, task_idx, num_episodes=10)
        results[task_result['task_name']] = task_result
        if verbose:
            print(f"  {task_result['task_name']}: {task_result['success_rate']*100:.1f}%")

    success_rates = [r['success_rate'] for r in results.values()]

    summary = {
        'per_task': results,
        'mean_success': np.mean(success_rates),
        'std_success': np.std(success_rates),
        'min_success': np.min(success_rates),
        'max_success': np.max(success_rates),
    }

    if verbose:
        print(f"\nMean Success: {summary['mean_success']*100:.1f}% ± {summary['std_success']*100:.1f}%")

    return summary


def eval_forward_transfer(
    suite_a: str = "libero_spatial",
    suite_b: str = "libero_object",
    verbose: bool = True,
) -> Dict:
    """Protocol B: Forward transfer evaluation."""
    if verbose:
        print("=" * 70)
        print(f"PROTOCOL B: Forward Transfer ({suite_a} → {suite_b})")
        print("=" * 70)

    adapter_a = LIBEROAdapter(suite_a)
    adapter_b = LIBEROAdapter(suite_b)

    # === Pretrained ===
    if verbose:
        print("\n--- Pretrained + Transfer ---")

    model_pretrained = LanguageConditionedPSA(
        obs_dim=10, action_dim=4, num_neurons=128, language_dim=32, readout_hidden=64,
    )

    if verbose:
        print("Pretraining on Suite A...")
    train_on_suite(model_pretrained, adapter_a, num_epochs=30, verbose=False)

    # Freeze PSA
    for param in model_pretrained.psa.parameters():
        param.requires_grad = False

    if verbose:
        print("Adapting to Suite B (readout only)...")
    train_on_suite(model_pretrained, adapter_b, num_epochs=15, verbose=False)

    pretrained_results = []
    for task_idx in range(adapter_b.num_tasks):
        r = adapter_b.evaluate(model_pretrained, task_idx)
        pretrained_results.append(r['success_rate'])

    # === From Scratch ===
    if verbose:
        print("\n--- From Scratch ---")

    model_scratch = LanguageConditionedPSA(
        obs_dim=10, action_dim=4, num_neurons=128, language_dim=32, readout_hidden=64,
    )

    if verbose:
        print("Training on Suite B from scratch...")
    train_on_suite(model_scratch, adapter_b, num_epochs=30, verbose=False)

    scratch_results = []
    for task_idx in range(adapter_b.num_tasks):
        r = adapter_b.evaluate(model_scratch, task_idx)
        scratch_results.append(r['success_rate'])

    summary = {
        'pretrained_mean': np.mean(pretrained_results),
        'scratch_mean': np.mean(scratch_results),
        'transfer_benefit': np.mean(pretrained_results) - np.mean(scratch_results),
    }

    if verbose:
        print(f"\nPretrained + Transfer: {summary['pretrained_mean']*100:.1f}%")
        print(f"From Scratch: {summary['scratch_mean']*100:.1f}%")
        print(f"Transfer Benefit: {summary['transfer_benefit']*100:.1f}%")

    return summary


def eval_forgetting(
    suites: List[str] = ["libero_spatial", "libero_object", "libero_goal"],
    verbose: bool = True,
) -> Dict:
    """Protocol C: Catastrophic forgetting evaluation."""
    if verbose:
        print("=" * 70)
        print(f"PROTOCOL C: Forgetting ({' → '.join(suites)})")
        print("=" * 70)

    adapters = [LIBEROAdapter(s) for s in suites]

    model = LanguageConditionedPSA(
        obs_dim=10, action_dim=4, num_neurons=128, language_dim=32, readout_hidden=64,
    )

    results = {'psa': {}}

    # Train sequentially, evaluate on first suite after each
    for i, adapter in enumerate(adapters):
        if verbose:
            print(f"\nTraining on Suite {i+1} ({suites[i]})...")
        train_on_suite(model, adapter, num_epochs=20, verbose=False)

        # Evaluate on Suite A
        suite_a_success = []
        for task_idx in range(adapters[0].num_tasks):
            r = adapters[0].evaluate(model, task_idx)
            suite_a_success.append(r['success_rate'])

        results['psa'][f'after_{suites[i]}'] = np.mean(suite_a_success)
        if verbose:
            print(f"  Suite A success after {suites[i]}: {np.mean(suite_a_success)*100:.1f}%")

    results['psa']['forgetting'] = results['psa'][f'after_{suites[0]}'] - results['psa'][f'after_{suites[-1]}']

    if verbose:
        print(f"\nPSA Forgetting: {results['psa']['forgetting']*100:.1f}%")

    return results


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='LIBERO Evaluation for PSA-SNN')
    parser.add_argument('--eval', type=str, default='all',
                        choices=['all', 'suite', 'transfer', 'forgetting'],
                        help='Evaluation protocol to run')
    parser.add_argument('--suite', type=str, default='libero_spatial',
                        help='Suite for single-suite evaluation')
    parser.add_argument('--suites', nargs='+', default=['libero_spatial', 'libero_object', 'libero_goal'],
                        help='Suites for forgetting evaluation')
    parser.add_argument('--output', type=str, default='results',
                        help='Output directory for results')

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    results = {}

    if args.eval in ['all', 'suite']:
        results['suite_level'] = eval_suite_level(args.suite)

    if args.eval in ['all', 'transfer']:
        results['forward_transfer'] = eval_forward_transfer()

    if args.eval in ['all', 'forgetting']:
        results['forgetting'] = eval_forgetting(args.suites)

    # Save results
    output_file = os.path.join(args.output, f'libero_results_{timestamp}.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)

    print(f"\nResults saved to {output_file}")


if __name__ == '__main__':
    main()
