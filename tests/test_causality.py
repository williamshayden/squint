from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from conftest import reseal_variant
from squint_rl.episode import Episode
from squint_rl.synthetic import make_synthetic_episode
from test_env import _env_types, make_env


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
