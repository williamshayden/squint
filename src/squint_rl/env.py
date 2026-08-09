from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from .budget import BudgetConfig, TokenBucket
from .episode import Episode
from .reward import MatchCounts, RewardState
from .tracker import FloatArray, Observation, ObservationScales, TrackBatch, Tracker

SKIP = 0
RUN_DETECTOR = 1

Info = dict[str, bool | float | int]


class SquintEnv(gym.Env[Observation, int]):
    """A causal replay environment for compute-budgeted tracking policies."""

    def __init__(
        self,
        *,
        episode: Episode,
        tracker: Tracker,
        budget: BudgetConfig,
        observation_scales: ObservationScales,
    ) -> None:
        normalized_refill_rate = budget.refill_ms_per_s / (episode.fps * budget.reserve_ms)
        if normalized_refill_rate > 1.0:
            raise ValueError("normalized refill rate must not exceed 1.0")
        self.metadata = {"render_modes": []}
        self.episode = episode
        self.tracker = tracker
        self.bucket = TokenBucket(budget)
        self.scales = observation_scales
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Dict(
            {
                "scene_change": gym.spaces.Box(0.0, 1.0, (3, 3), np.float32),
                "tracker_state": gym.spaces.Box(0.0, 1.0, (6,), np.float32),
                "compute_budget": gym.spaces.Box(
                    low=np.array([-np.inf, 0.0, 0.0, 0.0, 0.0], np.float32),
                    high=np.ones(5, np.float32),
                    dtype=np.float32,
                ),
            }
        )
        self._reward = RewardState()
        self.action_history: list[int] = []
        self.track_history: list[TrackBatch] = []
        self._frame_index = 0
        self._current_timestamp_s = 0.0
        self._last_detector_timestamp_s: float | None = None
        self._previous_applied_action = SKIP
        self._detector_calls = 0
        self._started = False
        self._terminated = False

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Observation, dict[str, Any]]:
        super().reset(seed=seed)
        del options
        self.tracker.reset()
        self._reward.reset()
        self.action_history.clear()
        self.track_history.clear()
        self._frame_index = 0
        self._current_timestamp_s = self._timestamp_at(self._frame_index)
        self._last_detector_timestamp_s = None
        self._previous_applied_action = SKIP
        self._detector_calls = 0
        self._terminated = False
        self._started = True
        self.bucket.reset(timestamp_s=self._current_timestamp_s)
        return self._observation(self._scene_change_at(self._frame_index)), {}

    def step(self, action: int) -> tuple[Observation, float, bool, bool, Info]:
        self._validate_step(action)
        requested_action = int(action)
        denied = requested_action == RUN_DETECTOR and not self.bucket.affordable
        applied_action = SKIP if denied else requested_action
        frame = self.episode.frame(self._frame_index)
        detections = frame.detections if applied_action == RUN_DETECTOR else None
        tracks = self.tracker.step(detections, frame.timestamp_s)

        charged_ms = 0.0
        if applied_action == RUN_DETECTOR:
            charged_ms = frame.detector_latency_ms
            self.bucket.charge(charged_ms)
            self._last_detector_timestamp_s = frame.timestamp_s
            self._detector_calls += 1

        reward, counts = self._reward.score(frame.ground_truth, tracks)
        self.action_history.append(applied_action)
        self.track_history.append(tracks)
        self._previous_applied_action = applied_action

        self._terminated = self._frame_index == self.episode.frame_count - 1
        if self._terminated:
            observation = self._observation(None)
        else:
            self._frame_index += 1
            self._current_timestamp_s = self._timestamp_at(self._frame_index)
            self.bucket.refill(timestamp_s=self._current_timestamp_s)
            observation = self._observation(self._scene_change_at(self._frame_index))

        return observation, reward, self._terminated, False, self._info(
            requested_action=requested_action,
            applied_action=applied_action,
            denied=denied,
            charged_ms=charged_ms,
            counts=counts,
        )

    def _validate_step(self, action: int) -> None:
        if not self._started:
            raise RuntimeError("reset must be called before step")
        if self._terminated:
            raise RuntimeError("episode is terminated; call reset before step")
        if (
            isinstance(action, (bool, np.bool_))
            or not isinstance(action, (int, np.integer))
            or int(action) not in (SKIP, RUN_DETECTOR)
        ):
            raise ValueError("action must be SKIP (0) or RUN_DETECTOR (1)")

    def _timestamp_at(self, frame_index: int) -> float:
        return float(self.episode.arrays["timestamps_s"][frame_index])

    def _scene_change_at(self, frame_index: int) -> FloatArray:
        return np.array(self.episode.arrays["scene_change"][frame_index], dtype=np.float32, copy=True)

    def _tracker_vector(self) -> FloatArray:
        summary = self.tracker.summary()
        denominator = max(1, summary.active_tracks)
        return np.array(
            [
                np.clip(summary.active_tracks / self.scales.active_tracks, 0.0, 1.0),
                np.clip(summary.confirmed_tracks / denominator, 0.0, 1.0),
                np.clip(summary.stale_tracks / denominator, 0.0, 1.0),
                np.clip(summary.mean_age_s / self.scales.age_s, 0.0, 1.0),
                np.clip(summary.mean_motion_px_s / self.scales.motion_px_s, 0.0, 1.0),
                np.clip(summary.mean_confidence, 0.0, 1.0),
            ],
            dtype=np.float32,
        )

    def _budget_vector(self) -> FloatArray:
        if self._last_detector_timestamp_s is None:
            elapsed = self.scales.time_since_detector_s
        else:
            elapsed = self._current_timestamp_s - self._last_detector_timestamp_s
        return np.array(
            [
                self.bucket.normalized_balance,
                self.bucket.config.refill_ms_per_s
                / (self.episode.fps * self.bucket.config.reserve_ms),
                float(self.bucket.affordable),
                np.clip(elapsed / self.scales.time_since_detector_s, 0.0, 1.0),
                float(self._previous_applied_action),
            ],
            dtype=np.float32,
        )

    def _observation(self, scene_change: FloatArray | None) -> Observation:
        return {
            "scene_change": (
                np.zeros((3, 3), dtype=np.float32)
                if scene_change is None
                else np.array(scene_change, dtype=np.float32, copy=True)
            ),
            "tracker_state": self._tracker_vector(),
            "compute_budget": self._budget_vector(),
        }

    def _info(
        self,
        *,
        requested_action: int,
        applied_action: int,
        denied: bool,
        charged_ms: float,
        counts: MatchCounts,
    ) -> Info:
        return {
            "requested_action": requested_action,
            "applied_action": applied_action,
            "denied": denied,
            "balance_ms": self.bucket.balance_ms,
            "charged_ms": charged_ms,
            "detector_calls": self._detector_calls,
            "matches": counts.matches,
            "false_positives": counts.false_positives,
            "false_negatives": counts.false_negatives,
            "identity_switches": counts.identity_switches,
            "localization_error": counts.localization_error,
        }
