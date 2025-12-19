#!/usr/bin/env python3
"""
PSA-SNN Evaluation on Real LIBERO Environment with Video Recording.

Runs all 4 evaluations with actual LIBERO environments:
1. Suite-level success (10 tasks)
2. Forward transfer (pretrain A → adapt B)
3. Forgetting (sequential A → B → C)
4. Sample efficiency (1/5/10/25 demos)

Usage:
    export LIBERO_FOLDER=/home/yubo/libero_data
    xvfb-run -a python run_libero_real.py
"""

import os
import sys
import json
import time
import h5py
import imageio
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# Set up virtual display for headless rendering
os.environ["MUJOCO_GL"] = "egl"

# Add LIBERO to path
sys.path.insert(0, os.path.expanduser("~/LIBERO"))
os.environ.setdefault("LIBERO_FOLDER", os.path.expanduser("~/libero_data"))

from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv

from psa.language_conditioned import LanguageConditionedPSA

# Create output directories
OUTPUT_DIR = Path("results")
VIDEO_DIR = OUTPUT_DIR / "videos"
OUTPUT_DIR.mkdir(exist_ok=True)
VIDEO_DIR.mkdir(exist_ok=True)

print("=" * 70)
print("PSA-SNN LIBERO EVALUATION (Real Environment)")
print("=" * 70)
print(f"Start time: {datetime.now()}")
print(f"Output directory: {OUTPUT_DIR}")
print()


# ============================================================================
# LIBERO Interface
# ============================================================================

def get_task_embedding(task_idx: int, num_tasks: int = 10) -> torch.Tensor:
    """Create structured task embedding."""
    embed = torch.zeros(32)
    # Structured embedding based on task
    embed[task_idx % 16] = 1.0
    embed[16 + (task_idx * 7) % 16] = 1.0
    embed = embed + torch.randn(32) * 0.1
    return embed


def extract_state(obs: dict) -> torch.Tensor:
    """Extract robot state from LIBERO observation dict."""
    # LIBERO provides various observation keys
    # We use robot proprioception as primary state
    if 'robot0_proprio-state' in obs:
        state = obs['robot0_proprio-state']
    elif 'robot0_eef_pos' in obs:
        # Combine end-effector pos, quat, gripper
        state = np.concatenate([
            obs.get('robot0_eef_pos', np.zeros(3)),
            obs.get('robot0_eef_quat', np.zeros(4)),
            obs.get('robot0_gripper_qpos', np.zeros(2)),
        ])
    else:
        # Fallback: concatenate all available state info
        state = np.concatenate([v.flatten() for k, v in sorted(obs.items())
                               if 'state' in k or 'pos' in k][:10])

    # Ensure consistent size (pad or truncate to 10)
    if len(state) > 10:
        state = state[:10]
    elif len(state) < 10:
        state = np.pad(state, (0, 10 - len(state)))

    return torch.tensor(state, dtype=torch.float32)


def load_demos_from_hdf5(demo_dir: str, task_name: str, max_demos: int = 10) -> List[List[Tuple]]:
    """Load demonstrations from LIBERO HDF5 files."""
    demos = []
    demo_path = Path(demo_dir) / f"{task_name}_demo.hdf5"

    if not demo_path.exists():
        # Try alternate naming
        demo_files = list(Path(demo_dir).glob(f"*{task_name}*.hdf5"))
        if demo_files:
            demo_path = demo_files[0]
        else:
            print(f"Warning: No demos found for {task_name}")
            return demos

    try:
        with h5py.File(demo_path, 'r') as f:
            data = f['data']
            demo_keys = sorted([k for k in data.keys() if k.startswith('demo')])[:max_demos]

            for demo_key in demo_keys:
                demo = data[demo_key]
                obs_data = demo['obs']
                actions = demo['actions'][:]

                trajectory = []
                for i in range(len(actions)):
                    # Extract state at timestep i
                    state_dict = {k: obs_data[k][i] for k in obs_data.keys()}
                    state = extract_state(state_dict)
                    action = torch.tensor(actions[i], dtype=torch.float32)

                    # Next state
                    if i + 1 < len(actions):
                        next_state_dict = {k: obs_data[k][i+1] for k in obs_data.keys()}
                        next_state = extract_state(next_state_dict)
                    else:
                        next_state = state.clone()

                    trajectory.append((state, action, next_state))

                if len(trajectory) > 0:
                    demos.append(trajectory)

    except Exception as e:
        print(f"Error loading demos from {demo_path}: {e}")

    return demos


def create_env(task, render: bool = False) -> OffScreenRenderEnv:
    """Create LIBERO environment for a task."""
    env = OffScreenRenderEnv(
        bddl_file=task.bddl_file,
        camera_heights=256 if render else 128,
        camera_widths=256 if render else 128,
        has_renderer=False,
        has_offscreen_renderer=True,
    )
    return env


def evaluate_episode(
    model: LanguageConditionedPSA,
    env: OffScreenRenderEnv,
    task_embed: torch.Tensor,
    max_steps: int = 300,
    record_video: bool = False,
) -> Tuple[bool, float, List[np.ndarray], List[torch.Tensor]]:
    """Run one evaluation episode."""
    model.eval()
    model.reset()
    model.set_task_embedding(task_embed)

    obs = env.reset()
    state = extract_state(obs)

    frames = []
    actions = []
    total_reward = 0
    success = False

    for step in range(max_steps):
        # Record frame if requested
        if record_video:
            frame = env.sim.render(
                camera_name="agentview",
                width=256,
                height=256,
            )
            frames.append(frame)

        # Get action from model
        with torch.no_grad():
            action, _, _ = model.forward(state)

        actions.append(action.clone())

        # Execute action
        obs, reward, done, info = env.step(action.numpy())
        state = extract_state(obs)
        total_reward += reward

        # Check success
        if info.get('success', False):
            success = True
            break

        if done:
            break

    model.train()
    return success, total_reward, frames, actions


def compute_jerk(actions: List[torch.Tensor]) -> float:
    """Compute action jerk (smoothness metric)."""
    if len(actions) < 3:
        return 0.0

    jerks = []
    for i in range(2, len(actions)):
        jerk = (actions[i] - 2*actions[i-1] + actions[i-2]).norm().item()
        jerks.append(jerk)

    return np.mean(jerks)


def save_video(frames: List[np.ndarray], path: str, fps: int = 30):
    """Save frames as video."""
    if len(frames) == 0:
        return

    # Flip frames vertically (MuJoCo renders upside down)
    frames = [np.flipud(f) for f in frames]

    imageio.mimsave(path, frames, fps=fps)
    print(f"  Saved video: {path}")


# ============================================================================
# Training Functions
# ============================================================================

def train_on_suite(
    model: LanguageConditionedPSA,
    suite_name: str,
    demos_per_task: int = 10,
    num_epochs: int = 30,
    verbose: bool = True,
) -> Dict:
    """Train model on all tasks in a suite using real demos."""
    bm = benchmark.get_benchmark(suite_name)
    task_names = bm.get_task_names()
    num_tasks = len(task_names)

    if verbose:
        print(f"Training on {suite_name} ({num_tasks} tasks, {demos_per_task} demos each)...")

    # Load demos
    demo_dir = os.environ.get("LIBERO_FOLDER", "~/libero_data")
    all_demos = {}

    for task_idx in range(num_tasks):
        task = bm.get_task(task_idx)
        task_name = task_names[task_idx]

        # Try to load real demos
        demos = load_demos_from_hdf5(demo_dir, task_name, demos_per_task)

        if len(demos) == 0:
            # Generate synthetic demos as fallback
            print(f"  Using synthetic demos for {task_name}")
            demos = generate_synthetic_demos(task_idx, demos_per_task)
        else:
            print(f"  Loaded {len(demos)} demos for {task_name}")

        embed = get_task_embedding(task_idx, num_tasks)
        all_demos[task_idx] = {'demos': demos, 'embedding': embed, 'name': task_name}

    # Train
    train_history = []
    for epoch in range(num_epochs):
        total_error = 0
        total_steps = 0

        task_order = list(range(num_tasks))
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

        avg_error = total_error / max(total_steps, 1)
        train_history.append(avg_error)

        if verbose and (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{num_epochs}: MSE = {avg_error:.4f}")

    model.consolidate()
    return {'train_history': train_history, 'task_names': task_names}


def generate_synthetic_demos(task_idx: int, num_demos: int = 10, demo_length: int = 50) -> List[List[Tuple]]:
    """Generate synthetic demos as fallback."""
    demos = []
    np.random.seed(task_idx * 1000)

    for _ in range(num_demos):
        trajectory = []
        state = torch.randn(10) * 0.3

        target = torch.zeros(10)
        target[task_idx % 10] = 0.5

        for _ in range(demo_length):
            error = target[:4] - state[:4]
            action = 0.3 * error + torch.randn(4) * 0.02
            action = torch.clamp(action, -1, 1)

            next_state = state.clone()
            next_state[:4] = state[:4] + action * 0.1
            next_state[4:] = state[4:] * 0.95

            trajectory.append((state.clone(), action.clone(), next_state.clone()))
            state = next_state

        demos.append(trajectory)

    return demos


def evaluate_on_suite(
    model: LanguageConditionedPSA,
    suite_name: str,
    num_episodes: int = 10,
    record_videos: bool = True,
    video_prefix: str = "",
    verbose: bool = True,
) -> Dict:
    """Evaluate model on all tasks using real LIBERO environments."""
    bm = benchmark.get_benchmark(suite_name)
    task_names = bm.get_task_names()
    num_tasks = len(task_names)

    if verbose:
        print(f"\nEvaluating on {suite_name} ({num_tasks} tasks)...")

    results = {}
    all_jerks = []

    for task_idx in range(num_tasks):
        task = bm.get_task(task_idx)
        task_name = task_names[task_idx]
        task_embed = get_task_embedding(task_idx, num_tasks)

        # Create environment
        env = create_env(task, render=record_videos)

        successes = 0
        task_jerks = []
        task_rewards = []

        for ep in range(num_episodes):
            # Record video for first episode of each task
            record = record_videos and ep == 0

            success, reward, frames, actions = evaluate_episode(
                model, env, task_embed,
                max_steps=300,
                record_video=record
            )

            if success:
                successes += 1
            task_rewards.append(reward)

            jerk = compute_jerk(actions)
            task_jerks.append(jerk)

            # Save video
            if record and len(frames) > 0:
                video_path = VIDEO_DIR / f"{video_prefix}{task_name}_ep{ep}.mp4"
                save_video(frames, str(video_path))

        env.close()

        results[task_name] = {
            'success_rate': successes / num_episodes,
            'mean_reward': np.mean(task_rewards),
            'mean_jerk': np.mean(task_jerks),
        }
        all_jerks.extend(task_jerks)

        if verbose:
            print(f"  {task_name}: {successes}/{num_episodes} success ({results[task_name]['success_rate']*100:.0f}%)")

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
# Evaluation Protocols
# ============================================================================

def eval_suite_level(suite_name: str = "libero_10") -> Dict:
    """Protocol A: Suite-level success."""
    print("\n" + "=" * 70)
    print(f"PROTOCOL A: Suite-Level Success ({suite_name})")
    print("=" * 70)

    model = LanguageConditionedPSA(
        obs_dim=10, action_dim=7, num_neurons=128, language_dim=32, readout_hidden=64,
    )

    train_results = train_on_suite(model, suite_name, demos_per_task=10, num_epochs=30)
    eval_results = evaluate_on_suite(model, suite_name, num_episodes=10,
                                     record_videos=True, video_prefix="suite_")

    print(f"\n--- Results ---")
    print(f"Mean Success: {eval_results['mean_success']*100:.1f}% ± {eval_results['std_success']*100:.1f}%")
    print(f"Mean Jerk: {eval_results['mean_jerk']:.4f}")

    return {'train': train_results, 'eval': eval_results}


def eval_forward_transfer(suite_a: str = "libero_spatial", suite_b: str = "libero_object") -> Dict:
    """Protocol B: Forward transfer."""
    print("\n" + "=" * 70)
    print(f"PROTOCOL B: Forward Transfer ({suite_a} → {suite_b})")
    print("=" * 70)

    # Pretrained
    print("\n--- Pretrained + Transfer ---")
    model_pre = LanguageConditionedPSA(
        obs_dim=10, action_dim=7, num_neurons=128, language_dim=32, readout_hidden=64,
    )

    print("Pretraining on Suite A...")
    train_on_suite(model_pre, suite_a, num_epochs=30, verbose=False)

    # Freeze PSA, only train readout
    for param in model_pre.psa.parameters():
        param.requires_grad = False

    print("Adapting to Suite B (readout only)...")
    train_on_suite(model_pre, suite_b, num_epochs=15, verbose=False)

    pre_results = evaluate_on_suite(model_pre, suite_b, num_episodes=5,
                                    record_videos=True, video_prefix="transfer_pre_")

    # From scratch
    print("\n--- From Scratch ---")
    model_scratch = LanguageConditionedPSA(
        obs_dim=10, action_dim=7, num_neurons=128, language_dim=32, readout_hidden=64,
    )

    print("Training on Suite B from scratch...")
    train_on_suite(model_scratch, suite_b, num_epochs=30, verbose=False)

    scratch_results = evaluate_on_suite(model_scratch, suite_b, num_episodes=5,
                                        record_videos=True, video_prefix="transfer_scratch_")

    print(f"\n--- Results ---")
    print(f"Pretrained + Transfer: {pre_results['mean_success']*100:.1f}%")
    print(f"From Scratch: {scratch_results['mean_success']*100:.1f}%")
    print(f"Transfer Benefit: {(pre_results['mean_success'] - scratch_results['mean_success'])*100:.1f}%")

    return {
        'pretrained': pre_results,
        'scratch': scratch_results,
        'transfer_benefit': pre_results['mean_success'] - scratch_results['mean_success'],
    }


def eval_forgetting(suites: List[str] = ["libero_spatial", "libero_object", "libero_goal"]) -> Dict:
    """Protocol C: Forgetting."""
    print("\n" + "=" * 70)
    print(f"PROTOCOL C: Forgetting ({' → '.join(suites)})")
    print("=" * 70)

    model = LanguageConditionedPSA(
        obs_dim=10, action_dim=7, num_neurons=128, language_dim=32, readout_hidden=64,
    )

    results = {'psa': {}}

    for i, suite in enumerate(suites):
        print(f"\nTraining on Suite {i+1} ({suite})...")
        train_on_suite(model, suite, num_epochs=20, verbose=False)

        # Evaluate on Suite A
        suite_a_results = evaluate_on_suite(model, suites[0], num_episodes=5,
                                           record_videos=(i == len(suites)-1),
                                           video_prefix=f"forget_after{i}_",
                                           verbose=False)
        results['psa'][f'after_{suite}'] = suite_a_results['mean_success']
        print(f"  Suite A success: {suite_a_results['mean_success']*100:.1f}%")

    results['psa']['forgetting'] = results['psa'][f'after_{suites[0]}'] - results['psa'][f'after_{suites[-1]}']

    print(f"\n--- Results ---")
    print(f"PSA Forgetting: {results['psa']['forgetting']*100:.1f}%")

    return results


def eval_sample_efficiency(suite_name: str = "libero_10") -> Dict:
    """Protocol D: Sample efficiency."""
    print("\n" + "=" * 70)
    print(f"PROTOCOL D: Sample Efficiency ({suite_name})")
    print("=" * 70)

    demo_counts = [1, 5, 10, 25]
    results = {'psa': {}}

    for num_demos in demo_counts:
        print(f"\n--- {num_demos} demos per task ---")

        model = LanguageConditionedPSA(
            obs_dim=10, action_dim=7, num_neurons=128, language_dim=32, readout_hidden=64,
        )

        train_on_suite(model, suite_name, demos_per_task=num_demos, num_epochs=30, verbose=False)

        eval_results = evaluate_on_suite(model, suite_name, num_episodes=5,
                                        record_videos=(num_demos == 1),
                                        video_prefix=f"sample{num_demos}_",
                                        verbose=False)

        results['psa'][num_demos] = eval_results['mean_success']
        print(f"  PSA Success: {eval_results['mean_success']*100:.1f}%")

    print(f"\n--- Results ---")
    print(f"{'Demos':<10} {'Success':>10}")
    print("-" * 25)
    for num_demos in demo_counts:
        print(f"{num_demos:<10} {results['psa'][num_demos]*100:>9.1f}%")

    return results


# ============================================================================
# Main
# ============================================================================

def main():
    results = {}
    start_time = time.time()

    # Check which suites are available
    available_suites = list(benchmark.get_benchmark_dict().keys())
    print(f"Available suites: {available_suites}")

    # Use libero_10 as primary (smaller, faster)
    primary_suite = "libero_10" if "libero_10" in available_suites else available_suites[0]

    # Run evaluations
    try:
        results['suite_level'] = eval_suite_level(primary_suite)
    except Exception as e:
        print(f"Suite-level eval failed: {e}")
        results['suite_level'] = {'error': str(e)}

    try:
        results['forward_transfer'] = eval_forward_transfer("libero_spatial", "libero_object")
    except Exception as e:
        print(f"Forward transfer eval failed: {e}")
        results['forward_transfer'] = {'error': str(e)}

    try:
        results['forgetting'] = eval_forgetting(["libero_spatial", "libero_object", "libero_goal"])
    except Exception as e:
        print(f"Forgetting eval failed: {e}")
        results['forgetting'] = {'error': str(e)}

    try:
        results['sample_efficiency'] = eval_sample_efficiency(primary_suite)
    except Exception as e:
        print(f"Sample efficiency eval failed: {e}")
        results['sample_efficiency'] = {'error': str(e)}

    # Summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    if 'eval' in results.get('suite_level', {}):
        print(f"Suite-Level Mean Success: {results['suite_level']['eval']['mean_success']*100:.1f}%")

    if 'transfer_benefit' in results.get('forward_transfer', {}):
        print(f"Forward Transfer Benefit: {results['forward_transfer']['transfer_benefit']*100:.1f}%")

    if 'psa' in results.get('forgetting', {}):
        print(f"PSA Forgetting: {results['forgetting']['psa'].get('forgetting', 0)*100:.1f}%")

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed/60:.1f} minutes")
    print(f"End time: {datetime.now()}")

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = OUTPUT_DIR / f"libero_real_eval_{timestamp}.json"

    def convert(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    with open(output_file, 'w') as f:
        json.dump(convert(results), f, indent=2)

    print(f"\nResults saved to {output_file}")
    print(f"Videos saved to {VIDEO_DIR}")


if __name__ == "__main__":
    main()
