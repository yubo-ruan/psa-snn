"""
LIBERO-specific evaluation metrics for PSA.

Key metrics:
1. Language-conditioned prediction quality
2. Surprise maps (where is the agent confused?)
3. Assembly-task mutual information
4. Lifelong stability metrics
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class PredictionQualityMetrics:
    """Metrics for prediction quality."""
    mean_prediction_error: float
    prediction_error_by_modality: Dict[str, float]  # vision, proprio, action
    temporal_consistency: float  # Are predictions stable over time?
    language_conditioning_effect: float  # How much does language help?


@dataclass
class SurpriseMapMetrics:
    """Metrics for surprise analysis."""
    mean_surprise: float
    surprise_variance: float
    high_surprise_ratio: float  # Fraction of time with surprise > threshold
    surprise_by_phase: Dict[str, float]  # reach, grasp, place phases


@dataclass
class AssemblyMetrics:
    """Metrics for assembly quality."""
    num_assemblies: int
    mean_assembly_size: float
    assembly_stability: float  # How often do assemblies reform?
    task_assembly_mi: float  # Mutual information between assemblies and tasks


@dataclass
class LifelongMetrics:
    """Metrics for lifelong learning stability."""
    forward_transfer: float  # Does learning new tasks help old ones?
    backward_transfer: float  # Does learning old tasks help new ones?
    forgetting: float  # How much performance degrades on old tasks?
    plasticity: float  # How quickly can the agent learn new tasks?


class PredictionQualityTracker:
    """
    Track language-conditioned prediction quality.

    Key question: Does the agent predict better when given the right instruction?
    """

    def __init__(self, modalities: List[str] = ["vision", "proprio"]):
        self.modalities = modalities
        self.reset()

    def reset(self):
        """Reset tracking."""
        self.errors = {m: [] for m in self.modalities}
        self.errors_with_language = {m: [] for m in self.modalities}
        self.errors_without_language = {m: [] for m in self.modalities}
        self.temporal_errors = []

    def update(
        self,
        predicted: Dict[str, torch.Tensor],
        actual: Dict[str, torch.Tensor],
        has_language: bool = True,
    ):
        """
        Update prediction error tracking.

        Args:
            predicted: Predicted values by modality
            actual: Actual values by modality
            has_language: Whether language conditioning was used
        """
        for modality in self.modalities:
            if modality in predicted and modality in actual:
                error = (predicted[modality] - actual[modality]).abs().mean().item()
                self.errors[modality].append(error)

                if has_language:
                    self.errors_with_language[modality].append(error)
                else:
                    self.errors_without_language[modality].append(error)

    def update_temporal(
        self,
        predictions: List[torch.Tensor],
        actuals: List[torch.Tensor],
    ):
        """
        Track temporal consistency of predictions.

        Good predictions should be smooth over time.
        """
        if len(predictions) < 2:
            return

        # Compute variance of prediction changes
        pred_changes = []
        for i in range(1, len(predictions)):
            change = (predictions[i] - predictions[i-1]).abs().mean().item()
            pred_changes.append(change)

        actual_changes = []
        for i in range(1, len(actuals)):
            change = (actuals[i] - actuals[i-1]).abs().mean().item()
            actual_changes.append(change)

        # Consistency = correlation between predicted and actual changes
        if len(pred_changes) > 0 and len(actual_changes) > 0:
            pred_var = np.var(pred_changes)
            actual_var = np.var(actual_changes)
            if actual_var > 1e-6:
                self.temporal_errors.append(pred_var / actual_var)

    def get_metrics(self) -> PredictionQualityMetrics:
        """Get current metrics."""
        # Mean error across modalities
        all_errors = []
        for m in self.modalities:
            all_errors.extend(self.errors[m])
        mean_error = np.mean(all_errors) if all_errors else 0.0

        # Error by modality
        error_by_modality = {}
        for m in self.modalities:
            error_by_modality[m] = np.mean(self.errors[m]) if self.errors[m] else 0.0

        # Temporal consistency
        temporal_consistency = np.mean(self.temporal_errors) if self.temporal_errors else 1.0

        # Language conditioning effect
        with_lang = []
        without_lang = []
        for m in self.modalities:
            with_lang.extend(self.errors_with_language[m])
            without_lang.extend(self.errors_without_language[m])

        if with_lang and without_lang:
            # Positive = language helps
            language_effect = np.mean(without_lang) - np.mean(with_lang)
        else:
            language_effect = 0.0

        return PredictionQualityMetrics(
            mean_prediction_error=mean_error,
            prediction_error_by_modality=error_by_modality,
            temporal_consistency=temporal_consistency,
            language_conditioning_effect=language_effect,
        )


class SurpriseMapTracker:
    """
    Track surprise patterns across episodes.

    Key questions:
    - Where is the agent confused?
    - Does surprise correlate with task difficulty?
    - Do certain phases (reach, grasp, place) have higher surprise?
    """

    def __init__(
        self,
        surprise_threshold: float = 0.7,
        phase_boundaries: Dict[str, Tuple[float, float]] = None,
    ):
        self.surprise_threshold = surprise_threshold
        self.phase_boundaries = phase_boundaries or {
            "reach": (0.0, 0.3),
            "grasp": (0.3, 0.6),
            "place": (0.6, 1.0),
        }
        self.reset()

    def reset(self):
        """Reset tracking."""
        self.surprises = []
        self.surprise_by_phase = defaultdict(list)
        self.episode_progress = 0.0
        self.episode_surprises = []

    def start_episode(self):
        """Start new episode."""
        self.episode_progress = 0.0
        self.episode_surprises = []

    def update(self, surprise: float, progress: float = None):
        """
        Update surprise tracking.

        Args:
            surprise: Current surprise value [0, 1]
            progress: Episode progress [0, 1] (if known)
        """
        self.surprises.append(surprise)

        if progress is not None:
            self.episode_progress = progress

        self.episode_surprises.append(surprise)

        # Assign to phase
        for phase, (start, end) in self.phase_boundaries.items():
            if start <= self.episode_progress < end:
                self.surprise_by_phase[phase].append(surprise)
                break

    def end_episode(self):
        """End current episode."""
        pass

    def get_metrics(self) -> SurpriseMapMetrics:
        """Get current metrics."""
        if not self.surprises:
            return SurpriseMapMetrics(
                mean_surprise=0.5,
                surprise_variance=0.0,
                high_surprise_ratio=0.0,
                surprise_by_phase={},
            )

        surprises = np.array(self.surprises)

        # By phase
        phase_means = {}
        for phase, values in self.surprise_by_phase.items():
            phase_means[phase] = np.mean(values) if values else 0.5

        return SurpriseMapMetrics(
            mean_surprise=surprises.mean(),
            surprise_variance=surprises.var(),
            high_surprise_ratio=(surprises > self.surprise_threshold).mean(),
            surprise_by_phase=phase_means,
        )

    def get_surprise_heatmap(
        self,
        grid_size: int = 10,
    ) -> np.ndarray:
        """
        Get 2D surprise heatmap over (time, task_progress).

        Useful for visualization.
        """
        # For now, just return 1D histogram over progress
        heatmap = np.zeros(grid_size)
        counts = np.zeros(grid_size)

        for i, (surprise, progress) in enumerate(
            zip(self.episode_surprises,
                np.linspace(0, 1, len(self.episode_surprises)))
        ):
            bin_idx = min(int(progress * grid_size), grid_size - 1)
            heatmap[bin_idx] += surprise
            counts[bin_idx] += 1

        # Average
        heatmap = np.divide(heatmap, counts, where=counts > 0)

        return heatmap


class AssemblyQualityTracker:
    """
    Track assembly formation and quality.

    Key questions:
    - Do assemblies correspond to meaningful task structures?
    - Are assemblies stable over time?
    - Do different tasks activate different assemblies?
    """

    def __init__(self, max_assemblies: int = 100):
        self.max_assemblies = max_assemblies
        self.reset()

    def reset(self):
        """Reset tracking."""
        self.assembly_activations = defaultdict(list)  # assembly_id -> [activation_times]
        self.task_assembly_counts = defaultdict(lambda: defaultdict(int))  # task -> assembly -> count
        self.assembly_sizes = {}
        self.step = 0

    def update(
        self,
        active_assemblies: Dict[int, float],
        assembly_sizes: Dict[int, int],
        current_task: Optional[str] = None,
    ):
        """
        Update assembly tracking.

        Args:
            active_assemblies: Assembly ID -> activation level
            assembly_sizes: Assembly ID -> number of neurons
            current_task: Current task name (for MI computation)
        """
        self.step += 1

        for assembly_id, activation in active_assemblies.items():
            if activation > 0.3:  # Threshold for "active"
                self.assembly_activations[assembly_id].append(self.step)

                if current_task:
                    self.task_assembly_counts[current_task][assembly_id] += 1

        self.assembly_sizes.update(assembly_sizes)

    def compute_assembly_stability(self) -> float:
        """
        Compute how stable assemblies are over time.

        Stable assemblies should activate in regular patterns.
        """
        stabilities = []

        for assembly_id, times in self.assembly_activations.items():
            if len(times) < 5:
                continue

            # Compute inter-activation intervals
            intervals = np.diff(times)
            if len(intervals) > 0:
                # Low variance = stable
                cv = np.std(intervals) / (np.mean(intervals) + 1e-6)
                stability = 1.0 / (1.0 + cv)  # Higher is more stable
                stabilities.append(stability)

        return np.mean(stabilities) if stabilities else 0.5

    def compute_task_assembly_mi(self) -> float:
        """
        Compute mutual information between tasks and assemblies.

        High MI = assemblies are task-specific (good for transfer).
        Low MI = assemblies are task-agnostic (might be generic).
        """
        if not self.task_assembly_counts:
            return 0.0

        # Build joint distribution
        tasks = list(self.task_assembly_counts.keys())
        all_assemblies = set()
        for task_counts in self.task_assembly_counts.values():
            all_assemblies.update(task_counts.keys())
        assemblies = list(all_assemblies)

        if len(tasks) < 2 or len(assemblies) < 2:
            return 0.0

        # Joint counts
        joint = np.zeros((len(tasks), len(assemblies)))
        for i, task in enumerate(tasks):
            for j, assembly in enumerate(assemblies):
                joint[i, j] = self.task_assembly_counts[task][assembly]

        # Normalize
        joint = joint / (joint.sum() + 1e-6)

        # Marginals
        p_task = joint.sum(axis=1)
        p_assembly = joint.sum(axis=0)

        # MI
        mi = 0.0
        for i in range(len(tasks)):
            for j in range(len(assemblies)):
                if joint[i, j] > 1e-10:
                    mi += joint[i, j] * np.log(
                        joint[i, j] / (p_task[i] * p_assembly[j] + 1e-10) + 1e-10
                    )

        return max(0.0, mi)

    def get_metrics(self) -> AssemblyMetrics:
        """Get current metrics."""
        num_assemblies = len(self.assembly_activations)

        sizes = list(self.assembly_sizes.values())
        mean_size = np.mean(sizes) if sizes else 0.0

        stability = self.compute_assembly_stability()
        mi = self.compute_task_assembly_mi()

        return AssemblyMetrics(
            num_assemblies=num_assemblies,
            mean_assembly_size=mean_size,
            assembly_stability=stability,
            task_assembly_mi=mi,
        )


class LifelongLearningTracker:
    """
    Track lifelong learning metrics across tasks.

    Key questions:
    - Does learning new tasks help with old ones (forward transfer)?
    - Does the agent forget old tasks?
    - How plastic is the agent (can it learn new things)?
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset tracking."""
        self.task_performances = defaultdict(list)  # task -> [performance over time]
        self.task_learning_curves = defaultdict(list)  # task -> [(step, performance)]
        self.current_step = 0

    def update(
        self,
        task: str,
        performance: float,
        is_training: bool = False,
    ):
        """
        Update with task performance.

        Args:
            task: Task name
            performance: Performance metric (e.g., success rate)
            is_training: Whether this was a training episode
        """
        self.current_step += 1

        self.task_performances[task].append(performance)
        if is_training:
            self.task_learning_curves[task].append((self.current_step, performance))

    def compute_forward_transfer(self, task_order: List[str]) -> float:
        """
        Compute forward transfer: does learning early tasks help later tasks?

        Positive = early tasks help later ones learn faster.
        """
        if len(task_order) < 2:
            return 0.0

        transfers = []

        for i, task in enumerate(task_order[1:], 1):
            if task not in self.task_learning_curves:
                continue

            curve = self.task_learning_curves[task]
            if len(curve) < 5:
                continue

            # Learning speed = slope of first 5 points
            first_5 = [p for _, p in curve[:5]]
            if len(first_5) >= 2:
                slope = (first_5[-1] - first_5[0]) / len(first_5)

                # Compare to expected baseline (could be from random init)
                baseline_slope = 0.1  # Assume baseline
                transfer = slope - baseline_slope
                transfers.append(transfer)

        return np.mean(transfers) if transfers else 0.0

    def compute_backward_transfer(self, task_order: List[str]) -> float:
        """
        Compute backward transfer: does learning new tasks improve old ones?

        Positive = learning new tasks helps old tasks.
        Negative = catastrophic forgetting.
        """
        if len(task_order) < 2:
            return 0.0

        transfers = []

        for i, old_task in enumerate(task_order[:-1]):
            if old_task not in self.task_performances:
                continue

            perfs = self.task_performances[old_task]
            if len(perfs) < 2:
                continue

            # Compare first half to second half
            mid = len(perfs) // 2
            first_half = np.mean(perfs[:mid])
            second_half = np.mean(perfs[mid:])

            transfer = second_half - first_half
            transfers.append(transfer)

        return np.mean(transfers) if transfers else 0.0

    def compute_forgetting(self) -> float:
        """
        Compute forgetting: how much does performance degrade over time?

        0 = no forgetting
        Positive = some forgetting
        1 = complete forgetting
        """
        forgetting_scores = []

        for task, perfs in self.task_performances.items():
            if len(perfs) < 10:
                continue

            # Peak performance
            peak = max(perfs)

            # Final performance (average of last 10%)
            final_window = perfs[int(0.9 * len(perfs)):]
            final = np.mean(final_window)

            if peak > 0.1:  # Only count if task was learned
                forgetting = (peak - final) / peak
                forgetting_scores.append(max(0, forgetting))

        return np.mean(forgetting_scores) if forgetting_scores else 0.0

    def compute_plasticity(self) -> float:
        """
        Compute plasticity: how quickly can the agent learn new tasks?

        Higher = faster learning.
        """
        learning_speeds = []

        for task, curve in self.task_learning_curves.items():
            if len(curve) < 10:
                continue

            # Time to reach 80% of peak
            perfs = [p for _, p in curve]
            peak = max(perfs)
            threshold = 0.8 * peak

            time_to_threshold = len(perfs)  # Default: never reached
            for i, p in enumerate(perfs):
                if p >= threshold:
                    time_to_threshold = i
                    break

            # Normalize by task length
            speed = 1.0 - (time_to_threshold / len(perfs))
            learning_speeds.append(speed)

        return np.mean(learning_speeds) if learning_speeds else 0.5

    def get_metrics(self, task_order: Optional[List[str]] = None) -> LifelongMetrics:
        """Get current metrics."""
        if task_order is None:
            task_order = list(self.task_performances.keys())

        return LifelongMetrics(
            forward_transfer=self.compute_forward_transfer(task_order),
            backward_transfer=self.compute_backward_transfer(task_order),
            forgetting=self.compute_forgetting(),
            plasticity=self.compute_plasticity(),
        )


class PSAMetricsLogger:
    """
    Combined metrics logger for PSA experiments.

    Logs all metrics to a structured format.
    """

    def __init__(
        self,
        log_dir: Optional[str] = None,
        modalities: List[str] = ["vision", "proprio"],
    ):
        self.log_dir = log_dir

        # Initialize all trackers
        self.prediction_tracker = PredictionQualityTracker(modalities)
        self.surprise_tracker = SurpriseMapTracker()
        self.assembly_tracker = AssemblyQualityTracker()
        self.lifelong_tracker = LifelongLearningTracker()

        self.step = 0
        self.episode = 0
        self.metrics_history = []

    def start_episode(self, task: Optional[str] = None):
        """Start new episode."""
        self.episode += 1
        self.current_task = task
        self.surprise_tracker.start_episode()

    def end_episode(self, success: bool = False):
        """End current episode."""
        self.surprise_tracker.end_episode()

        if self.current_task:
            self.lifelong_tracker.update(
                self.current_task,
                1.0 if success else 0.0,
                is_training=True,
            )

    def update(
        self,
        predicted: Dict[str, torch.Tensor],
        actual: Dict[str, torch.Tensor],
        surprise: float,
        active_assemblies: Dict[int, float],
        assembly_sizes: Dict[int, int],
        has_language: bool = True,
        progress: float = None,
    ):
        """Update all trackers."""
        self.step += 1

        self.prediction_tracker.update(predicted, actual, has_language)
        self.surprise_tracker.update(surprise, progress)
        self.assembly_tracker.update(
            active_assemblies,
            assembly_sizes,
            self.current_task
        )

    def get_all_metrics(self) -> Dict:
        """Get all metrics as a dictionary."""
        return {
            "prediction": self.prediction_tracker.get_metrics().__dict__,
            "surprise": self.surprise_tracker.get_metrics().__dict__,
            "assembly": self.assembly_tracker.get_metrics().__dict__,
            "lifelong": self.lifelong_tracker.get_metrics().__dict__,
            "step": self.step,
            "episode": self.episode,
        }

    def log_summary(self) -> str:
        """Get human-readable summary."""
        metrics = self.get_all_metrics()

        lines = [
            "=" * 50,
            "PSA Metrics Summary",
            "=" * 50,
            f"Step: {metrics['step']}, Episode: {metrics['episode']}",
            "",
            "Prediction Quality:",
            f"  Mean Error: {metrics['prediction']['mean_prediction_error']:.4f}",
            f"  Language Effect: {metrics['prediction']['language_conditioning_effect']:.4f}",
            "",
            "Surprise:",
            f"  Mean: {metrics['surprise']['mean_surprise']:.4f}",
            f"  High Surprise Ratio: {metrics['surprise']['high_surprise_ratio']:.4f}",
            "",
            "Assemblies:",
            f"  Count: {metrics['assembly']['num_assemblies']}",
            f"  Stability: {metrics['assembly']['assembly_stability']:.4f}",
            f"  Task MI: {metrics['assembly']['task_assembly_mi']:.4f}",
            "",
            "Lifelong Learning:",
            f"  Forward Transfer: {metrics['lifelong']['forward_transfer']:.4f}",
            f"  Forgetting: {metrics['lifelong']['forgetting']:.4f}",
            f"  Plasticity: {metrics['lifelong']['plasticity']:.4f}",
            "=" * 50,
        ]

        return "\n".join(lines)
