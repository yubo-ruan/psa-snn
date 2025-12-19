"""
Active Inference Agent using PSA Network.

Actions are chosen to minimize expected future surprise.
"""

import torch
import torch.nn as nn
from typing import Tuple, List, Optional, Dict
import numpy as np

from .network import PSANetwork


class ActiveInferenceAgent(nn.Module):
    """
    Agent that uses PSA network for:
    1. World model (predicting next states)
    2. Action selection (minimize expected surprise)
    3. Exploration (seek informative states)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        layer_sizes: List[int] = [256, 128, 64],
        exploration_weight: float = 0.1,
        num_action_samples: int = 10,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.exploration_weight = exploration_weight
        self.num_action_samples = num_action_samples

        # PSA network for world model (state + action → next state)
        self.world_model = PSANetwork(
            input_dim=state_dim + action_dim,
            layer_sizes=layer_sizes,
        )

        # Decoder: top layer spikes → predicted next state
        self.state_decoder = nn.Sequential(
            nn.Linear(layer_sizes[-1], layer_sizes[-1]),
            nn.ReLU(),
            nn.Linear(layer_sizes[-1], state_dim),
        )

        # Action prior (for generating candidate actions)
        self.action_mean = nn.Parameter(torch.zeros(action_dim))
        self.action_logstd = nn.Parameter(torch.zeros(action_dim))

        # Track prediction history for uncertainty estimation
        self.register_buffer('prediction_errors', torch.zeros(100, state_dim))
        self.error_idx = 0

    def reset(self):
        """Reset world model."""
        self.world_model.reset()
        self.error_idx = 0

    def encode_state_action(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Encode state-action pair for world model."""
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)

        return torch.cat([state, action], dim=-1)

    def predict_next_state(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, float]:
        """
        Predict next state given current state and action.

        Args:
            state: Current state [batch, state_dim] or [state_dim]
            action: Action [batch, action_dim] or [action_dim]

        Returns:
            predicted_state: Predicted next state
            uncertainty: Prediction uncertainty
        """
        # Encode input
        x = self.encode_state_action(state, action)

        # Forward through world model
        top_spikes, info = self.world_model(x)

        # Decode to state
        predicted_state = self.state_decoder(top_spikes)

        # Estimate uncertainty
        uncertainty = info['surprise']

        return predicted_state.squeeze(0), uncertainty

    def compute_expected_surprise(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> float:
        """
        Compute expected surprise of taking action in state.

        Low surprise = predictable outcome = exploit
        High surprise = uncertain outcome = explore or avoid
        """
        _, uncertainty = self.predict_next_state(state, action)
        return uncertainty

    def compute_information_gain(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> float:
        """
        Estimate information gain from taking action.

        High info gain = action would teach us something new.
        """
        # Predict next state
        predicted_next, uncertainty = self.predict_next_state(state, action)

        # Information gain approximation:
        # How different is the predicted state from the current?
        if state.dim() == 1:
            state = state.unsqueeze(0)

        state_change = (predicted_next - state).abs().mean().item()

        # Combine with uncertainty (uncertain predictions are more informative)
        info_gain = state_change * uncertainty

        return info_gain

    def sample_actions(self, num_samples: int = None) -> torch.Tensor:
        """
        Sample candidate actions from prior.

        Args:
            num_samples: Number of actions to sample

        Returns:
            actions: Sampled actions [num_samples, action_dim]
        """
        if num_samples is None:
            num_samples = self.num_action_samples

        std = torch.exp(self.action_logstd)
        noise = torch.randn(num_samples, self.action_dim, device=self.action_mean.device)
        actions = self.action_mean + std * noise

        # Clip to reasonable range
        actions = torch.tanh(actions)

        return actions

    def select_action(
        self,
        state: torch.Tensor,
        explore: bool = True,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Select action using active inference.

        Minimize expected surprise, with optional exploration bonus.

        Args:
            state: Current state
            explore: Whether to include exploration bonus

        Returns:
            best_action: Selected action
            info: Dictionary with selection details
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)

        # Sample candidate actions
        candidate_actions = self.sample_actions()

        best_action = None
        best_score = float('inf')
        info = {
            'candidate_scores': [],
            'expected_surprises': [],
            'info_gains': [],
        }

        for i in range(candidate_actions.shape[0]):
            action = candidate_actions[i]

            # Compute expected surprise
            expected_surprise = self.compute_expected_surprise(state, action)
            info['expected_surprises'].append(expected_surprise)

            # Compute information gain
            info_gain = self.compute_information_gain(state, action)
            info['info_gains'].append(info_gain)

            # Score: minimize surprise, maximize info gain if exploring
            if explore:
                score = expected_surprise - self.exploration_weight * info_gain
            else:
                score = expected_surprise

            info['candidate_scores'].append(score)

            if score < best_score:
                best_score = score
                best_action = action

        info['selected_score'] = best_score

        return best_action, info

    def update_world_model(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor,
    ):
        """
        Update world model based on observed transition.

        Uses local PSA plasticity, not backprop.

        Args:
            state: Current state
            action: Action taken
            next_state: Resulting state
        """
        # Predict next state
        predicted_next, _ = self.predict_next_state(state, action)

        # Compute prediction error
        if next_state.dim() == 1:
            next_state = next_state.unsqueeze(0)
        error = next_state - predicted_next

        # Store error for uncertainty estimation
        self.prediction_errors[self.error_idx % 100] = error.squeeze(0)
        self.error_idx += 1

        # Update PSA network with local plasticity
        self.world_model.local_update()

        # Update action prior based on what worked
        # (Actions with low error should be more likely)
        error_magnitude = error.abs().mean().item()
        if error_magnitude < 0.5:  # Good prediction = reinforce action
            with torch.no_grad():
                self.action_mean.data = (
                    0.99 * self.action_mean.data +
                    0.01 * action
                )

    def get_world_model_summary(self) -> str:
        """Get summary of world model state."""
        lines = ["Active Inference Agent Summary"]
        lines.append(f"State dim: {self.state_dim}, Action dim: {self.action_dim}")
        lines.append(f"Exploration weight: {self.exploration_weight}")
        lines.append(f"Prediction errors tracked: {min(self.error_idx, 100)}")

        if self.error_idx > 0:
            recent_errors = self.prediction_errors[:min(self.error_idx, 100)]
            lines.append(f"Mean prediction error: {recent_errors.abs().mean():.4f}")

        lines.append("\n" + self.world_model.get_assembly_summary())

        return "\n".join(lines)


class ReachingAgent(ActiveInferenceAgent):
    """
    Specialized agent for robotic reaching tasks.

    State: joint angles + end effector position + target position
    Action: joint velocity commands
    """

    def __init__(
        self,
        num_joints: int = 7,
        ee_dim: int = 3,
        target_dim: int = 3,
        layer_sizes: List[int] = [128, 64, 32],
        exploration_weight: float = 0.1,
    ):
        state_dim = num_joints + ee_dim + target_dim
        action_dim = num_joints

        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            layer_sizes=layer_sizes,
            exploration_weight=exploration_weight,
        )

        self.num_joints = num_joints
        self.ee_dim = ee_dim
        self.target_dim = target_dim

    def compute_task_reward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor,
    ) -> float:
        """
        Compute task-specific reward (distance to target).

        Args:
            state: Current state
            action: Action taken
            next_state: Resulting state

        Returns:
            reward: Negative distance to target
        """
        # Extract end effector and target positions
        ee_start = self.num_joints
        ee_end = ee_start + self.ee_dim
        target_start = ee_end
        target_end = target_start + self.target_dim

        if next_state.dim() == 1:
            ee_pos = next_state[ee_start:ee_end]
            target_pos = next_state[target_start:target_end]
        else:
            ee_pos = next_state[0, ee_start:ee_end]
            target_pos = next_state[0, target_start:target_end]

        distance = torch.norm(ee_pos - target_pos).item()
        return -distance

    def select_action_with_goal(
        self,
        state: torch.Tensor,
        goal_weight: float = 1.0,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Select action considering both prediction and goal.

        Combines:
        - Minimize expected surprise (world model)
        - Minimize distance to goal (task)

        Args:
            state: Current state
            goal_weight: Weight on goal-directed behavior

        Returns:
            best_action: Selected action
            info: Selection details
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)

        candidate_actions = self.sample_actions()

        best_action = None
        best_score = float('inf')
        info = {'candidate_scores': [], 'goal_progress': []}

        # Get current distance to goal
        ee_start = self.num_joints
        ee_end = ee_start + self.ee_dim
        target_start = ee_end
        target_end = target_start + self.target_dim

        current_ee = state[0, ee_start:ee_end]
        target = state[0, target_start:target_end]
        current_distance = torch.norm(current_ee - target).item()

        for i in range(candidate_actions.shape[0]):
            action = candidate_actions[i]

            # Predict next state
            predicted_next, uncertainty = self.predict_next_state(state, action)

            # Predicted distance to goal
            if predicted_next.dim() == 1:
                predicted_ee = predicted_next[ee_start:ee_end]
            else:
                predicted_ee = predicted_next[0, ee_start:ee_end]

            predicted_distance = torch.norm(predicted_ee - target).item()

            # Goal progress (negative = getting closer)
            goal_progress = predicted_distance - current_distance
            info['goal_progress'].append(goal_progress)

            # Combined score
            score = uncertainty + goal_weight * goal_progress
            info['candidate_scores'].append(score)

            if score < best_score:
                best_score = score
                best_action = action

        return best_action, info
