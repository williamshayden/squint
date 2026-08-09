from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType
from typing import Any

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from conftest import RecordingTracker, reseal_variant
from squint_rl.budget import BudgetConfig
from squint_rl.episode import Episode
from squint_rl.synthetic import make_synthetic_episode
from squint_rl.tracker import (
    DetectionBatch,
    ObservationScales,
    TrackBatch,
)

INFO_KEYS = {
    "requested_action",
    "applied_action",
    "denied",
    "balance_ms",
    "charged_ms",
    "detector_calls",
    "matches",
    "false_positives",
    "false_negatives",
    "identity_switches",
    "localization_error",
}


def _env_module() -> ModuleType | None:
    try:
        return importlib.import_module("squint_rl.env")
    except ModuleNotFoundError:
        return None


def _env_types() -> tuple[type[Any], int, int]:
    module = _env_module()
    assert module is not None, "Task 6 causal Gym environment must exist"
    return module.SquintEnv, module.SKIP, module.RUN_DETECTOR


def make_env(
    episode: Episode,
    tracker: RecordingTracker | None = None,
    *,
    nominal_rate: float = 0.25,
    budget: BudgetConfig | None = None,
) -> tuple[Any, RecordingTracker]:
    squint_env, _skip, _run_detector = _env_types()
    recording_tracker = tracker if tracker is not None else RecordingTracker()
    return (
        squint_env(
            episode=episode,
            tracker=recording_tracker,
            budget=(
                budget
                if budget is not None
                else BudgetConfig.for_rate(
                    reserve_ms=10.0,
                    source_fps=episode.fps,
                    nominal_rate=nominal_rate,
                )
            ),
            observation_scales=ObservationScales(8.0, 5.0, 20.0, 5.0),
        ),
        recording_tracker,
    )


def assert_observation_contract(env: Any, observation: dict[str, np.ndarray]) -> None:
    assert set(observation) == {"scene_change", "tracker_state", "compute_budget"}
    assert observation["scene_change"].shape == (3, 3)
    assert observation["tracker_state"].shape == (6,)
    assert observation["compute_budget"].shape == (5,)
    assert all(value.dtype == np.float32 for value in observation.values())
    assert np.all((0.0 <= observation["scene_change"]) & (observation["scene_change"] <= 1.0))
    assert np.all((0.0 <= observation["tracker_state"]) & (observation["tracker_state"] <= 1.0))
    assert np.all(observation["compute_budget"][1:] >= 0.0)
    assert np.all(observation["compute_budget"][1:] <= 1.0)
    assert env.observation_space.contains(observation)


def test_environment_module_is_available() -> None:
    assert _env_module() is not None, "Task 6 causal Gym environment must exist"


def test_environment_passes_gymnasium_checker(sealed_episode: Episode) -> None:
    env, _tracker = make_env(sealed_episode)
    check_env(env, skip_render_check=True)


def test_registration_is_singleton_and_exports_squint_env(sealed_episode: Episode) -> None:
    squint_env, _skip, _run_detector = _env_types()
    import squint_rl

    assert squint_rl.SquintEnv is squint_env
    assert gym.spec("SquintReplay-v0").entry_point == "squint_rl.env:SquintEnv"
    registered = gym.make(
        "SquintReplay-v0",
        episode=sealed_episode,
        tracker=RecordingTracker(),
        budget=BudgetConfig.for_rate(
            reserve_ms=10.0,
            source_fps=sealed_episode.fps,
            nominal_rate=0.25,
        ),
        observation_scales=ObservationScales(8.0, 5.0, 20.0, 5.0),
    )
    assert isinstance(registered.unwrapped, squint_env)


def test_reset_returns_only_causal_frame_zero_observation(sealed_episode: Episode) -> None:
    env, tracker = make_env(sealed_episode)

    observation, info = env.reset(seed=7)

    assert info == {}
    assert tracker.reset_calls == 1
    assert_observation_contract(env, observation)
    np.testing.assert_array_equal(observation["scene_change"], np.ones((3, 3), np.float32))
    np.testing.assert_array_equal(observation["tracker_state"], np.zeros(6, np.float32))
    np.testing.assert_allclose(
        observation["compute_budget"], np.array([0.5, 0.25, 1.0, 1.0, 0.0], np.float32)
    )
    assert not {"frame_index", "duration", "remaining_frames", "detections", "cost", "ground_truth"} & set(
        observation
    )


def test_constructor_accepts_refill_rate_at_observation_upper_bound(
    sealed_episode: Episode,
) -> None:
    budget = BudgetConfig(
        reserve_ms=10.0,
        capacity_ms=20.0,
        refill_ms_per_s=sealed_episode.fps * 10.0,
    )
    env, _tracker = make_env(sealed_episode, budget=budget)

    observation, _info = env.reset(seed=0)

    assert observation["compute_budget"][1] == 1.0
    assert env.observation_space.contains(observation)


def test_constructor_rejects_refill_rate_above_observation_upper_bound(
    sealed_episode: Episode,
) -> None:
    budget = BudgetConfig(
        reserve_ms=10.0,
        capacity_ms=20.0,
        refill_ms_per_s=sealed_episode.fps * 10.0 + 0.01,
    )

    with pytest.raises(ValueError, match="normalized refill rate"):
        make_env(sealed_episode, budget=budget)


def test_admitted_run_steps_tracker_then_charges_actual_latency(sealed_episode: Episode) -> None:
    env, tracker = make_env(sealed_episode)
    _squint_env, _skip, run_detector = _env_types()
    env.reset(seed=0)

    observation, reward, terminated, truncated, info = env.step(run_detector)

    assert reward == 1.0
    assert not terminated
    assert not truncated
    assert info == {
        "requested_action": run_detector,
        "applied_action": run_detector,
        "denied": False,
        "balance_ms": 2.5,
        "charged_ms": 10.0,
        "detector_calls": 1,
        "matches": 1,
        "false_positives": 0,
        "false_negatives": 0,
        "identity_switches": 0,
        "localization_error": 0.0,
    }
    assert tracker.measurements[0] is not None and len(tracker.measurements[0]) == 1
    assert env.action_history == [run_detector]
    assert len(env.track_history) == 1
    assert len(env.track_history[0]) == 1
    assert_observation_contract(env, observation)
    np.testing.assert_allclose(
        observation["compute_budget"], np.array([0.125, 0.25, 0.0, 0.1, 1.0], np.float32)
    )


def test_unaffordable_run_is_applied_as_skip_without_measurement_or_charge(
    sealed_episode: Episode,
) -> None:
    env, tracker = make_env(sealed_episode)
    _squint_env, skip, run_detector = _env_types()
    env.reset(seed=0)
    env.step(run_detector)

    _observation, _reward, _terminated, truncated, info = env.step(run_detector)

    assert not truncated
    assert info["requested_action"] == run_detector
    assert info["applied_action"] == skip
    assert info["denied"] is True
    assert info["charged_ms"] == 0.0
    assert info["detector_calls"] == 1
    assert info["balance_ms"] == 5.0
    assert tracker.measurements[1] is None
    assert env.action_history == [run_detector, skip]


def test_skip_passes_none_and_empty_detector_result_stays_a_measurement(tmp_path: Path) -> None:
    base = Episode.open(
        make_synthetic_episode(
            tmp_path / "base", frame_count=3, fps=2.0, change_frames=(0,), latency_ms=10.0
        )
    )

    def remove_second_detection(arrays: dict[str, np.ndarray]) -> None:
        arrays["det_boxes_xyxy"] = np.delete(arrays["det_boxes_xyxy"], 1, axis=0)
        arrays["det_scores"] = np.delete(arrays["det_scores"], 1)
        arrays["det_class_ids"] = np.delete(arrays["det_class_ids"], 1)
        arrays["det_frame_offsets"][2:] -= 1

    episode = reseal_variant(base, tmp_path / "empty-second", remove_second_detection)
    env, tracker = make_env(episode, nominal_rate=1.0)
    _squint_env, skip, run_detector = _env_types()
    env.reset(seed=0)
    env.step(skip)
    _observation, _reward, _terminated, _truncated, info = env.step(run_detector)

    assert tracker.measurements[0] is None
    assert tracker.measurements[1] is not None
    assert len(tracker.measurements[1]) == 0
    assert info["applied_action"] == run_detector
    assert info["charged_ms"] == 10.0
    assert info["detector_calls"] == 1


def test_reset_clears_tracker_reward_and_histories(sealed_episode: Episode) -> None:
    class EpochTracker(RecordingTracker):
        def __init__(self) -> None:
            super().__init__()
            self.epoch = 0

        def reset(self) -> None:
            super().reset()
            self.epoch += 1

        def step(self, detections: DetectionBatch | None, timestamp_s: float) -> TrackBatch:
            tracks = super().step(detections, timestamp_s)
            if detections is None:
                return tracks
            self.last = TrackBatch(
                detections.boxes_xyxy,
                np.full(len(detections), self.epoch, dtype=np.int64),
                detections.class_ids,
                detections.scores,
            )
            return self.last

    tracker = EpochTracker()
    env, _tracker = make_env(sealed_episode, tracker)
    _squint_env, _skip, run_detector = _env_types()
    env.reset(seed=0)
    env.step(run_detector)
    reset_observation, reset_info = env.reset(seed=0)
    _observation, _reward, _terminated, _truncated, info = env.step(run_detector)

    assert reset_info == {}
    assert tracker.reset_calls == 2
    assert env.action_history == [run_detector]
    assert len(env.track_history) == 1
    assert info["identity_switches"] == 0
    assert_observation_contract(env, reset_observation)


def test_episode_terminates_after_every_frame_without_truncation(sealed_episode: Episode) -> None:
    env, _tracker = make_env(sealed_episode)
    _squint_env, skip, _run_detector = _env_types()
    env.reset(seed=0)

    for _frame_index in range(sealed_episode.frame_count - 1):
        _observation, _reward, terminated, truncated, _info = env.step(skip)
        assert not terminated
        assert not truncated
    observation, _reward, terminated, truncated, _info = env.step(skip)

    assert terminated
    assert not truncated
    np.testing.assert_array_equal(observation["scene_change"], np.zeros((3, 3), np.float32))
    assert_observation_contract(env, observation)


def test_invalid_actions_and_lifecycle_errors_do_not_mutate_state(sealed_episode: Episode) -> None:
    env, tracker = make_env(sealed_episode)
    control, control_tracker = make_env(sealed_episode)
    _squint_env, skip, run_detector = _env_types()

    with pytest.raises(RuntimeError, match="reset"):
        env.step(skip)
    assert env.action_history == []
    assert tracker.measurements == []

    env.reset(seed=0)
    control.reset(seed=0)
    for action in (-1, 2, 0.0, True, "run"):
        with pytest.raises(ValueError, match="action"):
            env.step(action)

    observation, reward, terminated, truncated, info = env.step(run_detector)
    control_observation, control_reward, control_terminated, control_truncated, control_info = control.step(
        run_detector
    )
    for key, value in observation.items():
        np.testing.assert_array_equal(value, control_observation[key])
    assert reward == control_reward
    assert terminated == control_terminated
    assert truncated == control_truncated
    assert info == control_info
    assert env.action_history == control.action_history
    assert len(env.track_history) == len(control.track_history)
    for tracks, control_tracks in zip(env.track_history, control.track_history, strict=True):
        np.testing.assert_array_equal(tracks.boxes_xyxy, control_tracks.boxes_xyxy)
        np.testing.assert_array_equal(tracks.track_ids, control_tracks.track_ids)
        np.testing.assert_array_equal(tracks.class_ids, control_tracks.class_ids)
        np.testing.assert_array_equal(tracks.scores, control_tracks.scores)
    assert len(tracker.measurements) == len(control_tracker.measurements)
    for measurement, control_measurement in zip(
        tracker.measurements, control_tracker.measurements, strict=True
    ):
        assert (measurement is None) == (control_measurement is None)
        if measurement is not None:
            assert control_measurement is not None
            np.testing.assert_array_equal(measurement.boxes_xyxy, control_measurement.boxes_xyxy)
            np.testing.assert_array_equal(measurement.scores, control_measurement.scores)
            np.testing.assert_array_equal(measurement.class_ids, control_measurement.class_ids)

    for _frame_index in range(sealed_episode.frame_count - 1):
        _observation, _reward, terminated, _truncated, _info = env.step(skip)
    assert terminated
    history = list(env.action_history)
    with pytest.raises(RuntimeError, match="terminated"):
        env.step(skip)
    assert env.action_history == history
    assert len(tracker.measurements) == sealed_episode.frame_count


def test_info_schema_is_exact_for_every_transition(sealed_episode: Episode) -> None:
    env, _tracker = make_env(sealed_episode)
    _squint_env, skip, _run_detector = _env_types()
    env.reset(seed=0)

    _observation, _reward, _terminated, _truncated, info = env.step(skip)

    assert set(info) == INFO_KEYS
    assert info["requested_action"] == skip
    assert info["applied_action"] == skip
    assert info["denied"] is False
    assert info["charged_ms"] == 0.0
    assert info["detector_calls"] == 0
