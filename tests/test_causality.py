from __future__ import annotations

import hashlib
import struct
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from conftest import reseal_variant
from squint_rl.episode import Episode
from squint_rl.synthetic import make_synthetic_episode
from test_env import _env_types, make_env

_OBSERVATION_PREFIX_DOMAIN = b"squint.causal-observation-prefix.v1\0"
_REQUESTED_ACTION_PREFIX_DOMAIN = b"squint.causal-requested-action-prefix.v1\0"
_APPLIED_ACTION_PREFIX_DOMAIN = b"squint.causal-applied-action-prefix.v1\0"
_OBSERVATION_KEYS = ("scene_change", "tracker_state", "compute_budget")
_OBSERVATION_SHAPES = {
    "scene_change": (3, 3),
    "tracker_state": (6,),
    "compute_budget": (5,),
}


def _observation_prefix_digest(observations: list[dict[str, np.ndarray]]) -> str:
    digest = hashlib.sha256(
        _OBSERVATION_PREFIX_DOMAIN + struct.pack(">Q", len(observations))
    )
    for observation in observations:
        assert tuple(observation) == _OBSERVATION_KEYS
        for key in _OBSERVATION_KEYS:
            value = observation[key]
            assert type(value) is np.ndarray
            assert value.dtype.kind == "f"
            assert value.dtype.itemsize == 4
            assert value.shape == _OBSERVATION_SHAPES[key]
            canonical_value = np.asarray(value, dtype="<f4", order="C")
            key_bytes = key.encode("utf-8")
            payload = canonical_value.tobytes(order="C")
            digest.update(struct.pack(">Q", len(key_bytes)))
            digest.update(key_bytes)
            digest.update(struct.pack(">Q", len(payload)))
            digest.update(payload)
    return digest.hexdigest()


def _action_prefix_digest(actions: list[int], domain: bytes) -> str:
    digest = hashlib.sha256(domain + struct.pack(">Q", len(actions)))
    for action in actions:
        assert type(action) is int
        assert action in (0, 1)
        digest.update(bytes((action,)))
    return digest.hexdigest()


def _requested_action_prefix_digest(actions: list[int]) -> str:
    return _action_prefix_digest(actions, _REQUESTED_ACTION_PREFIX_DOMAIN)


def _applied_action_prefix_digest(actions: list[int]) -> str:
    return _action_prefix_digest(actions, _APPLIED_ACTION_PREFIX_DOMAIN)


def test_canonical_observation_prefix_zero_fixture() -> None:
    zero_observation = {
        "scene_change": np.zeros((3, 3), dtype=np.float32),
        "tracker_state": np.zeros(6, dtype=np.float32),
        "compute_budget": np.zeros(5, dtype=np.float32),
    }
    assert _observation_prefix_digest([zero_observation]) == (
        "76b144cbc5dc5e7a000c1d1545cf3471f655ef290eff7a4710bdc62e1585c8d1"
    )


def test_observation_digest_canonicalizes_float32_layout_and_endianness() -> None:
    native_zero = {
        "scene_change": np.zeros((3, 3), dtype=np.float32),
        "tracker_state": np.zeros(6, dtype=np.float32),
        "compute_budget": np.zeros(5, dtype=np.float32),
    }
    equivalent_zero = {
        "scene_change": np.zeros((3, 3), dtype=np.float32, order="F"),
        "tracker_state": np.zeros(6, dtype=">f4"),
        "compute_budget": np.zeros(5, dtype=np.float32),
    }

    assert _observation_prefix_digest([equivalent_zero]) == _observation_prefix_digest(
        [native_zero]
    )


def test_observation_digest_rejects_non_float32_or_malformed_values() -> None:
    def zero_observation() -> dict[str, np.ndarray]:
        return {
            "scene_change": np.zeros((3, 3), dtype=np.float32),
            "tracker_state": np.zeros(6, dtype=np.float32),
            "compute_budget": np.zeros(5, dtype=np.float32),
        }

    invalid_observations = []
    float64_observation = zero_observation()
    float64_observation["scene_change"] = np.zeros((3, 3), dtype=np.float64)
    invalid_observations.append(float64_observation)
    object_observation = zero_observation()
    object_observation["tracker_state"] = np.zeros(6, dtype=object)
    invalid_observations.append(object_observation)
    wrong_shape_observation = zero_observation()
    wrong_shape_observation["compute_budget"] = np.zeros(4, dtype=np.float32)
    invalid_observations.append(wrong_shape_observation)
    wrong_type_observation = zero_observation()
    wrong_type_observation["scene_change"] = [[0.0] * 3] * 3  # type: ignore[assignment]
    invalid_observations.append(wrong_type_observation)

    for invalid_observation in invalid_observations:
        with pytest.raises(AssertionError):
            _observation_prefix_digest([invalid_observation])


def test_canonical_requested_action_prefix_111_fixture() -> None:
    assert _requested_action_prefix_digest([1, 1, 1]) == (
        "ecd13fa82b2adabebffd26d8195a44477b910de37ea3443deb12ad004e7d0950"
    )


def test_canonical_applied_action_prefix_100_fixture() -> None:
    assert _applied_action_prefix_digest([1, 0, 0]) == (
        "5515a8a354459fb606623f7a32295447cf3755499760b477ba420ceaa51caae3"
    )


@dataclass(frozen=True)
class _PrefixEvidence:
    observations: tuple[dict[str, np.ndarray], ...]
    requested_actions: tuple[int, ...]
    applied_actions: tuple[int, ...]
    rewards: tuple[float, ...]

    @property
    def observation_digest(self) -> str:
        return _observation_prefix_digest(list(self.observations))

    @property
    def requested_action_digest(self) -> str:
        return _requested_action_prefix_digest(list(self.requested_actions))

    @property
    def applied_action_digest(self) -> str:
        return _applied_action_prefix_digest(list(self.applied_actions))


def _rollout_prefix(
    env: Any, requested_actions: tuple[int, ...], *, t: int
) -> _PrefixEvidence:
    assert len(requested_actions) == t + 1
    observation, _reset_info = env.reset(seed=7)
    observations = [observation]
    recorded_requested_actions: list[int] = []
    applied_actions: list[int] = []
    rewards: list[float] = []
    for action_index, requested_action in enumerate(requested_actions):
        next_observation, reward, _terminated, _truncated, info = env.step(requested_action)
        recorded_requested_actions.append(info["requested_action"])
        applied_actions.append(info["applied_action"])
        rewards.append(reward)
        if action_index < t:
            observations.append(next_observation)
    assert len(observations) == t + 1
    return _PrefixEvidence(
        observations=tuple(observations),
        requested_actions=tuple(recorded_requested_actions),
        applied_actions=tuple(applied_actions),
        rewards=tuple(rewards),
    )


def _byte_identical(left: np.ndarray, right: np.ndarray) -> bool:
    return (
        left.dtype == right.dtype
        and left.shape == right.shape
        and left.nbytes == right.nbytes
        and left.tobytes(order="C") == right.tobytes(order="C")
    )


def test_numeric_equality_does_not_imply_byte_identity() -> None:
    float_values = np.array([1], dtype=np.float32)
    integer_values = np.array([1], dtype=np.int32)

    assert np.array_equal(float_values, integer_values)
    assert not _byte_identical(float_values, integer_values)


def _valid_gt_rows(arrays: dict[str, np.ndarray], frames: range) -> np.ndarray:
    rows: list[int] = []
    offsets = arrays["gt_frame_offsets"]
    for frame in frames:
        start, stop = (int(value) for value in offsets[frame : frame + 2])
        rows.extend(
            index for index in range(start, stop) if bool(arrays["gt_valid"][index])
        )
    return np.asarray(rows, dtype=np.int64)


def assert_same_observation(left: dict[str, np.ndarray], right: dict[str, np.ndarray]) -> None:
    assert set(left) == set(right)
    for key, value in left.items():
        np.testing.assert_array_equal(value, right[key])


def make_base_episode(tmp_path: Path) -> Episode:
    return Episode.open(
        make_synthetic_episode(
            tmp_path / "base", frame_count=5, fps=2.0, change_frames=(0,), latency_ms=10.0
        )
    )


@pytest.fixture
def paired_prefix_envs(tmp_path: Path) -> tuple[tuple[Any, Any, Episode, Episode, str, int | None], ...]:
    base = make_base_episode(tmp_path)
    t = 2

    def mutate_future_detections(arrays: dict[str, np.ndarray]) -> None:
        arrays["det_boxes_xyxy"][t + 1 :, (0, 2)] += 4.0

    def mutate_future_latency(arrays: dict[str, np.ndarray]) -> None:
        arrays["detector_latency_ms"][t + 1 :] = 17.0

    def mutate_future_scene(arrays: dict[str, np.ndarray]) -> None:
        arrays["scene_change"][t + 1 :] = 0.75

    def mutate_future_valid_ground_truth(arrays: dict[str, np.ndarray]) -> None:
        rows = _valid_gt_rows(arrays, range(t + 1, 5))
        arrays["gt_boxes_xyxy"][rows, (0, 2)] += 5.0

    def mutate_current_valid_ground_truth(arrays: dict[str, np.ndarray]) -> None:
        rows = _valid_gt_rows(arrays, range(t, t + 1))
        arrays["gt_boxes_xyxy"][rows, (0, 2)] += 50.0

    def mutate_episode_view_boundary_scene(arrays: dict[str, np.ndarray]) -> None:
        arrays["scene_change"][2] = 0.75

    specs: tuple[tuple[str, str, Callable[[dict[str, np.ndarray]], None], int | None, bool], ...] = (
        ("future-detections", "det_boxes_xyxy", mutate_future_detections, None, False),
        ("future-latency", "detector_latency_ms", mutate_future_latency, None, False),
        ("future-scene", "scene_change", mutate_future_scene, None, False),
        ("future-valid-ground-truth", "gt_boxes_xyxy", mutate_future_valid_ground_truth, None, False),
        ("current-valid-ground-truth", "gt_boxes_xyxy", mutate_current_valid_ground_truth, t, False),
        ("episode-view-boundary-scene", "scene_change", mutate_episode_view_boundary_scene, None, True),
    )
    cases: list[tuple[Any, Any, Episode, Episode, str, int | None]] = []
    for name, changed_array, mutate, reward_diff_at, use_view in specs:
        left_source = reseal_variant(base, tmp_path / f"{name}-left", lambda arrays: None)
        right_source = reseal_variant(base, tmp_path / f"{name}-right", mutate)
        assert left_source.content_sha256 != right_source.content_sha256
        assert set(left_source.arrays) == set(right_source.arrays)
        for array_name in left_source.arrays:
            left_array = left_source.arrays[array_name]
            right_array = right_source.arrays[array_name]
            assert left_array.dtype == right_array.dtype
            assert left_array.shape == right_array.shape
            assert left_array.nbytes == right_array.nbytes
            assert _byte_identical(left_array, right_array) is (array_name != changed_array)
        left_episode: Any = left_source.slice(2, 5) if use_view else left_source
        right_episode: Any = right_source.slice(2, 5) if use_view else right_source
        assert left_episode.content_sha256 != right_episode.content_sha256
        left, _left_tracker = make_env(left_episode)
        right, _right_tracker = make_env(right_episode)
        cases.append((left, right, left_source, right_source, changed_array, reward_diff_at))
    return tuple(cases)


def test_independent_mutations_preserve_canonical_prefixes(
    paired_prefix_envs: tuple[tuple[Any, Any, Episode, Episode, str, int | None], ...],
) -> None:
    t = 2
    requested_actions = (1, 1, 1)
    for left, right, _left_source, _right_source, _changed_array, reward_diff_at in paired_prefix_envs:
        left_evidence = _rollout_prefix(left, requested_actions, t=t)
        right_evidence = _rollout_prefix(right, requested_actions, t=t)
        assert left_evidence.requested_actions == (1, 1, 1)
        assert right_evidence.requested_actions == (1, 1, 1)
        assert left_evidence.applied_actions == (1, 0, 0)
        assert right_evidence.applied_actions == (1, 0, 0)
        assert left_evidence.observation_digest == right_evidence.observation_digest
        assert left_evidence.requested_action_digest == right_evidence.requested_action_digest
        assert left_evidence.applied_action_digest == right_evidence.applied_action_digest
        if reward_diff_at is None:
            assert left_evidence.rewards == right_evidence.rewards
        else:
            assert left_evidence.rewards[:-1] == right_evidence.rewards[:-1]
            assert left_evidence.rewards[-1] != right_evidence.rewards[-1]


def test_current_cost_is_hidden_until_an_admitted_run(tmp_path: Path) -> None:
    base = make_base_episode(tmp_path)

    def change_current_cost(arrays: dict[str, np.ndarray]) -> None:
        arrays["detector_latency_ms"][0] = 15.0

    ordinary = reseal_variant(base, tmp_path / "ordinary", lambda arrays: None)
    expensive = reseal_variant(base, tmp_path / "expensive", change_current_cost)
    left, _left_tracker = make_env(ordinary)
    right, _right_tracker = make_env(expensive)
    _squint_env, _skip, run_detector = _env_types()
    left_observation, _left_info = left.reset(seed=7)
    right_observation, _right_info = right.reset(seed=7)

    assert_same_observation(left_observation, right_observation)
    left_next, _left_reward, _left_terminated, _left_truncated, left_info = left.step(run_detector)
    right_next, _right_reward, _right_terminated, _right_truncated, right_info = right.step(run_detector)

    assert_same_observation(
        {key: value for key, value in left_next.items() if key != "compute_budget"},
        {key: value for key, value in right_next.items() if key != "compute_budget"},
    )
    assert left_next["compute_budget"][0] != right_next["compute_budget"][0]
    assert left_info["charged_ms"] == 10.0
    assert right_info["charged_ms"] == 15.0


def test_hidden_ground_truth_changes_reward_but_not_observation(tmp_path: Path) -> None:
    base = make_base_episode(tmp_path)

    def move_current_ground_truth(arrays: dict[str, np.ndarray]) -> None:
        arrays["gt_boxes_xyxy"][0, (0, 2)] += 50.0

    matching = reseal_variant(base, tmp_path / "matching", lambda arrays: None)
    changed_ground_truth = reseal_variant(base, tmp_path / "changed-ground-truth", move_current_ground_truth)
    left, _left_tracker = make_env(matching)
    right, _right_tracker = make_env(changed_ground_truth)
    _squint_env, _skip, run_detector = _env_types()
    left_observation, _left_info = left.reset(seed=7)
    right_observation, _right_info = right.reset(seed=7)

    assert_same_observation(left_observation, right_observation)
    left_next, left_reward, _left_terminated, _left_truncated, _left_step_info = left.step(run_detector)
    right_next, right_reward, _right_terminated, _right_truncated, _right_step_info = right.step(run_detector)

    assert left_reward != right_reward
    assert_same_observation(left_next, right_next)
