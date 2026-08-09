from __future__ import annotations

import hashlib
import struct
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
            assert value.dtype == np.dtype("<f4")
            assert value.shape == _OBSERVATION_SHAPES[key]
            assert value.flags.c_contiguous
            key_bytes = key.encode("utf-8")
            payload = value.tobytes(order="C")
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


def test_canonical_requested_action_prefix_111_fixture() -> None:
    assert _requested_action_prefix_digest([1, 1, 1]) == (
        "ecd13fa82b2adabebffd26d8195a44477b910de37ea3443deb12ad004e7d0950"
    )


def test_canonical_applied_action_prefix_100_fixture() -> None:
    assert _applied_action_prefix_digest([1, 0, 0]) == (
        "5515a8a354459fb606623f7a32295447cf3755499760b477ba420ceaa51caae3"
    )


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
def paired_future_envs(tmp_path: Path) -> tuple[Any, Any, int]:
    base = make_base_episode(tmp_path)
    split_frame = 2

    def mutate_future(arrays: dict[str, np.ndarray]) -> None:
        start = split_frame + 1
        arrays["scene_change"][start:] = 0.75
        arrays["detector_latency_ms"][start:] = 17.0
        arrays["det_boxes_xyxy"][start:, 0] += 4.0
        arrays["det_boxes_xyxy"][start:, 2] += 4.0
        arrays["gt_boxes_xyxy"][2 * start :: 2, 0] += 5.0
        arrays["gt_boxes_xyxy"][2 * start :: 2, 2] += 5.0

    unchanged = reseal_variant(base, tmp_path / "left", lambda arrays: None)
    changed = reseal_variant(base, tmp_path / "right", mutate_future)
    left, _left_tracker = make_env(unchanged)
    right, _right_tracker = make_env(changed)
    return left, right, split_frame


def test_future_changes_cannot_change_observations_through_split(
    paired_future_envs: tuple[Any, Any, int],
) -> None:
    left, right, split_frame = paired_future_envs
    _squint_env, skip, _run_detector = _env_types()
    left_observation, _left_info = left.reset(seed=7)
    right_observation, _right_info = right.reset(seed=7)

    for _frame in range(split_frame + 1):
        assert_same_observation(left_observation, right_observation)
        left_observation, _left_reward, _left_terminated, _left_truncated, _left_step_info = left.step(skip)
        right_observation, _right_reward, _right_terminated, _right_truncated, _right_step_info = right.step(skip)


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
