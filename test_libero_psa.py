"""
Test PSA on a simulated LIBERO pick-and-place task.

This uses synthetic data to validate the PSA architecture before
connecting to actual LIBERO environments.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, List
import time

from psa import (
    LIBEROPSAAgent,
    PSAMetricsLogger,
    FrozenVisionEncoder,
)


class SyntheticLIBEROEnv:
    """
    Synthetic LIBERO-like environment for testing PSA.

    Simulates a pick-and-place task:
    1. Reach phase: move arm toward object
    2. Grasp phase: close gripper on object
    3. Place phase: move to target location

    State: [arm_pos (3), gripper (1), object_pos (3), target_pos (3)]
    Action: [arm_delta (3), gripper_action (1)]
    """

    def __init__(
        self,
        episode_length: int = 50,
        noise_std: float = 0.02,
    ):
        self.episode_length = episode_length
        self.noise_std = noise_std

        # Task variations
        self.task_variations = [
            {"object": "red_cube", "target": "blue_plate", "instruction": "pick up the red cube and place it on the blue plate"},
            {"object": "green_ball", "target": "white_bowl", "instruction": "put the green ball in the white bowl"},
            {"object": "yellow_block", "target": "gray_tray", "instruction": "move the yellow block to the gray tray"},
        ]

        self.reset()

    def reset(self, task_idx: int = 0) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """
        Reset environment to initial state.

        Returns:
            image: Synthetic image [3, 128, 128]
            proprio: Proprioceptive state [7]
            instruction: Task instruction string
        """
        self.step_count = 0
        self.task = self.task_variations[task_idx % len(self.task_variations)]

        # Random initial positions
        self.arm_pos = torch.rand(3) * 0.5  # [0, 0.5] workspace
        self.gripper = torch.tensor([0.0])  # Open
        self.object_pos = torch.rand(3) * 0.3 + 0.3  # [0.3, 0.6]
        self.target_pos = torch.rand(3) * 0.3 + 0.5  # [0.5, 0.8]

        self.grasped = False
        self.done = False

        return self._get_obs()

    def _get_obs(self) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """Get current observation."""
        # Synthetic image (128x128 with colored blocks at object/target positions)
        image = self._render_synthetic_image()

        # Proprioception: arm position + gripper
        proprio = torch.cat([self.arm_pos, self.gripper, self.object_pos])

        return image, proprio, self.task["instruction"]

    def _render_synthetic_image(self) -> torch.Tensor:
        """Render a simple synthetic image."""
        image = torch.zeros(3, 128, 128)

        # Background
        image[0, :, :] = 0.2  # Slight red tint

        # Draw arm as white circle
        arm_x = int(self.arm_pos[0].item() * 100 + 14)
        arm_y = int(self.arm_pos[1].item() * 100 + 14)
        for dx in range(-5, 6):
            for dy in range(-5, 6):
                if dx*dx + dy*dy < 25:
                    x, y = min(127, max(0, arm_x + dx)), min(127, max(0, arm_y + dy))
                    image[:, y, x] = 1.0

        # Draw object as colored square
        obj_x = int(self.object_pos[0].item() * 100 + 14)
        obj_y = int(self.object_pos[1].item() * 100 + 14)
        for dx in range(-8, 9):
            for dy in range(-8, 9):
                x, y = min(127, max(0, obj_x + dx)), min(127, max(0, obj_y + dy))
                if "red" in self.task["object"]:
                    image[0, y, x] = 0.8
                elif "green" in self.task["object"]:
                    image[1, y, x] = 0.8
                else:
                    image[2, y, x] = 0.8

        # Draw target as outlined square
        tgt_x = int(self.target_pos[0].item() * 100 + 14)
        tgt_y = int(self.target_pos[1].item() * 100 + 14)
        for dx in range(-10, 11):
            for dy in range(-10, 11):
                if abs(dx) > 7 or abs(dy) > 7:  # Border only
                    x, y = min(127, max(0, tgt_x + dx)), min(127, max(0, tgt_y + dy))
                    image[1, y, x] = 0.6
                    image[2, y, x] = 0.6

        return image

    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, str, float, bool, Dict]:
        """
        Take action in environment.

        Args:
            action: [arm_delta (3), gripper_action (1)]

        Returns:
            image, proprio, instruction, reward, done, info
        """
        self.step_count += 1

        # Parse action
        arm_delta = action[:3] * 0.1  # Scale down
        gripper_action = action[3:4]

        # Add noise
        arm_delta = arm_delta + torch.randn(3) * self.noise_std

        # Update arm position
        self.arm_pos = torch.clamp(self.arm_pos + arm_delta, 0, 1)

        # Update gripper
        if gripper_action > 0.5:
            self.gripper = torch.tensor([1.0])  # Close
        elif gripper_action < -0.5:
            self.gripper = torch.tensor([0.0])  # Open

        # Check grasping
        arm_to_obj = torch.norm(self.arm_pos - self.object_pos)
        if arm_to_obj < 0.1 and self.gripper > 0.5 and not self.grasped:
            self.grasped = True

        # Move object with arm if grasped
        if self.grasped:
            self.object_pos = self.arm_pos.clone()

        # Compute reward
        reward = 0.0
        info = {"phase": "reach"}

        if not self.grasped:
            # Reward for getting close to object
            reward = -arm_to_obj.item()
            info["phase"] = "reach"
        else:
            # Reward for getting close to target
            obj_to_target = torch.norm(self.object_pos - self.target_pos)
            reward = -obj_to_target.item()
            info["phase"] = "place"

            if obj_to_target < 0.1:
                reward += 10.0  # Big bonus for success
                info["success"] = True
                self.done = True

        # Episode length limit
        if self.step_count >= self.episode_length:
            self.done = True

        info["progress"] = self.step_count / self.episode_length

        return *self._get_obs(), reward, self.done, info


def get_dummy_language_embedding(instruction: str) -> torch.Tensor:
    """
    Get a dummy language embedding.

    In practice, use CLIP or sentence-transformers.
    """
    # Hash the instruction to get a consistent embedding
    hash_val = hash(instruction)
    torch.manual_seed(abs(hash_val) % (2**31))
    embedding = torch.randn(384)  # Match language_dim
    return embedding


def run_episode(
    agent: LIBEROPSAAgent,
    env: SyntheticLIBEROEnv,
    metrics: PSAMetricsLogger,
    task_idx: int = 0,
    train: bool = True,
) -> Dict:
    """Run a single episode."""
    # Reset
    agent.reset()
    image, proprio, instruction = env.reset(task_idx)

    # Set language instruction
    lang_embedding = get_dummy_language_embedding(instruction)
    agent.set_instruction(lang_embedding)

    # Start episode tracking
    metrics.start_episode(task=f"task_{task_idx}")

    total_reward = 0.0
    episode_length = 0
    success = False

    # Store for temporal tracking
    predictions = []
    actuals = []

    while True:
        episode_length += 1

        # Get action from agent
        action, info = agent(image, proprio)

        # Take step
        next_image, next_proprio, _, reward, done, step_info = env.step(action)

        # Track metrics
        predicted = {"vision": agent.vision(image), "proprio": proprio}
        actual = {"vision": agent.vision(next_image), "proprio": next_proprio}

        # Get assembly info
        active_assemblies = {}
        assembly_sizes = {}
        for i, module in enumerate(agent.psa.psa_modules):
            for assembly_id, assembly in module.fast_layer.assembly_detectors.items() if hasattr(module.fast_layer, 'assembly_detectors') else []:
                active_assemblies[assembly_id] = 0.5  # Placeholder
                assembly_sizes[assembly_id] = 5  # Placeholder

        metrics.update(
            predicted=predicted,
            actual=actual,
            surprise=info.get('fast_surprise', 0.5) if 'fast_surprise' in info else 0.5,
            active_assemblies=active_assemblies,
            assembly_sizes=assembly_sizes,
            progress=step_info.get("progress", 0.0),
        )

        # Store for temporal
        predictions.append(predicted["proprio"])
        actuals.append(actual["proprio"])

        # Update agent with local plasticity
        if train:
            agent.update()

        total_reward += reward
        success = step_info.get("success", False)

        if done:
            break

        image, proprio = next_image, next_proprio

    # End episode
    metrics.end_episode(success=success)

    # Track temporal consistency
    metrics.prediction_tracker.update_temporal(predictions, actuals)

    return {
        "total_reward": total_reward,
        "episode_length": episode_length,
        "success": success,
        "task_idx": task_idx,
    }


def main():
    print("=" * 60)
    print("PSA LIBERO Test - Synthetic Pick-and-Place")
    print("=" * 60)

    # Create agent
    print("\nCreating agent...")
    agent = LIBEROPSAAgent(
        image_size=(128, 128),
        proprio_dim=7,  # arm (3) + gripper (1) + object (3)
        action_dim=4,   # arm delta (3) + gripper (1)
        language_dim=384,
        num_modules=4,  # Fewer modules for testing
        neurons_per_module=32,  # Smaller for testing
    )

    print(f"Agent created with {sum(p.numel() for p in agent.parameters()):,} parameters")

    # Create environment
    env = SyntheticLIBEROEnv(episode_length=50)

    # Create metrics logger
    metrics = PSAMetricsLogger(modalities=["vision", "proprio"])

    # Training loop
    num_episodes = 50
    eval_every = 10

    print(f"\nRunning {num_episodes} episodes...")
    print("-" * 60)

    episode_results = []
    start_time = time.time()

    for ep in range(num_episodes):
        # Rotate through tasks
        task_idx = ep % 3

        # Run episode
        result = run_episode(
            agent, env, metrics,
            task_idx=task_idx,
            train=True,
        )
        episode_results.append(result)

        # Print progress
        if (ep + 1) % eval_every == 0:
            # Recent results
            recent = episode_results[-eval_every:]
            avg_reward = np.mean([r["total_reward"] for r in recent])
            success_rate = np.mean([r["success"] for r in recent])

            elapsed = time.time() - start_time
            eps_per_sec = (ep + 1) / elapsed

            print(f"Episode {ep+1:3d} | Reward: {avg_reward:7.2f} | Success: {success_rate:.0%} | "
                  f"Speed: {eps_per_sec:.1f} ep/s")

    print("-" * 60)
    print("\nTraining complete!")

    # Print final metrics
    print("\n" + metrics.log_summary())

    # Test calibration
    print("\nHomeostatic Calibration Status:")
    print(agent.psa.calibration.get_summary())

    # Test on each task
    print("\nPer-Task Evaluation:")
    print("-" * 40)

    for task_idx in range(3):
        task_results = [r for r in episode_results if r["task_idx"] == task_idx]
        if task_results:
            avg_reward = np.mean([r["total_reward"] for r in task_results[-5:]])
            success = np.mean([r["success"] for r in task_results[-5:]])
            print(f"Task {task_idx}: Reward={avg_reward:7.2f}, Success={success:.0%}")

    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
